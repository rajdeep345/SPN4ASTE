import warnings
warnings.filterwarnings("ignore")
import logging
formatter = logging.Formatter('%(message)s')
import json
import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold
from utils.metric import metric, num_metric, overlap_metric

def setup_logger(name, log_file, level=logging.DEBUG):
	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger


def get_tuples(pred):
	pred_tuples = dict()
	for sent_idx in pred:
		prediction = list(set([(ele.pred_rel, ele.head_start_index, ele.head_end_index, ele.tail_start_index, ele.tail_end_index) \
			for ele in pred[sent_idx]]))
		pred_tuples[sent_idx] = prediction
	return pred_tuples


class Trainer(nn.Module):
	def __init__(self, model, data, args):
		super().__init__()
		self.args = args
		self.model = model
		self.data = data

		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		component = ['encoder', 'decoder']
		grouped_params = [
			{
				'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
				'weight_decay': args.weight_decay,
				'lr': args.encoder_lr
			},
			{
				'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
				'weight_decay': 0.0,
				'lr': args.encoder_lr
			},
			{
				'params': [p for n, p in self.model.named_parameters() if
						   not any(nd in n for nd in no_decay) and component[1] in n],
				'weight_decay': args.weight_decay,
				'lr': args.decoder_lr
			},
			{
				'params': [p for n, p in self.model.named_parameters() if
						   any(nd in n for nd in no_decay) and component[1] in n],
				'weight_decay': 0.0,
				'lr': args.decoder_lr
			}
		]
		if args.optimizer == 'Adam':
			self.optimizer = optim.Adam(grouped_params)
		elif args.optimizer == 'AdamW':
			self.optimizer = AdamW(grouped_params)
		else:
			raise Exception("Invalid optimizer.")
		if args.use_gpu:
			self.cuda()

	def train_model(self):
		best_test_f1 = 0
		best_val_f1  = 0
		train_loader = self.data.train_loader
		train_num = len(train_loader)
		batch_size = self.args.batch_size
		total_batch = train_num // batch_size + 1
		# result = self.eval_model(self.data.test_loader)
		for epoch in range(self.args.max_epoch):
			# Train
			self.model.train()
			self.model.zero_grad()
			self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
			print("=== Epoch %d train ===" % epoch, flush=True)
			avg_loss = AverageMeter()
			random.shuffle(train_loader)
			for batch_id in range(total_batch):
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size
				if end > train_num:
					end = train_num
				train_instance = train_loader[start:end]
				# print([ele[0] for ele in train_instance])
				if not train_instance:
					continue
				input_ids, attention_mask, targets, _ = self.model.batchify(train_instance)
				loss, _ = self.model(input_ids, attention_mask, targets)
				avg_loss.update(loss.item(), 1)
				# Optimize
				loss.backward()
				if self.args.max_grad_norm != 0:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
				if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
					self.optimizer.step()
					self.model.zero_grad()
				if batch_id % 100 == 0 and batch_id != 0:
					print("     Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)
			gc.collect()
			torch.cuda.empty_cache()
			
			# Test
			print("=== Epoch %d Test ===" % epoch, flush=True)
			result, prediction_test, gold_test = self.eval_model(self.data.test_loader)
			prec_test = result['precision']
			recall_test = result['recall']
			f1_test = result['f1']
			if f1_test > best_test_f1:
				print("\033[94m Achieved Best Result on Test Set in this epoch.\n \033[0m", flush=True)
				best_test_f1 = f1_test
				best_test_epoch = epoch
				# torch.save({'state_dict': self.model.state_dict()}, self.args.generated_param_directory + " %s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.dataset_name, epoch, result['f1']))
			
			# Validation
			print("=== Epoch %d Validation ===" % epoch)
			result_val, prediction_val, gold_val = self.eval_model(self.data.valid_loader)
			f1_val = result_val['f1']           
			if f1_val > best_val_f1:
				print("\033[94m Achieved Best Result on Dev Set in this epoch.\033[0m", flush=True)
				# torch.save({'state_dict': self.model.state_dict()}, self.args.generated_param_directory + " %s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.dataset_name, epoch, f1_val))
				best_val_f1 = f1_val
				best_val_epoch = epoch
				best_val_gold = gold_val
				best_val_pred = prediction_val

				best_saved_test_prec = prec_test
				best_saved_test_recall = recall_test
				best_saved_test_f1 = f1_test
				best_saved_test_gold = gold_test
				best_saved_test_pred = prediction_test

			gc.collect()
			torch.cuda.empty_cache()
		
		print("\033[94m Best result on test set was %f achieved at epoch %d. \033[0m" % (best_test_f1, best_test_epoch), flush=True)
		print("\033[94m Best result on dev set was %f achieved at epoch %d.\n \033[0m" % (best_val_f1, best_val_epoch), flush=True)
		print("\033[94m Corresponding result on test set - Precision: %f, Recall: %f, F1: %f \n \033[0m" % (best_saved_test_prec, best_saved_test_recall, best_saved_test_f1), flush=True)
		
		if self.args.crossType == 0:
			devLogger = setup_logger('dev_log', f'./logs/{self.args.dataset_name}/in-domain/dev_results_{self.args.random_seed}.log')
			testLogger = setup_logger('test_log', f'./logs/{self.args.dataset_name}/in-domain/test_results_{self.args.random_seed}.log')
		elif self.args.crossType == 1:
			devLogger = setup_logger('dev_log', f'./logs/{self.args.dataset_name}/cross1/dev_results_{self.args.random_seed}.log')
			testLogger = setup_logger('test_log', f'./logs/{self.args.dataset_name}/cross1/test_results_{self.args.random_seed}.log')
		else:
			devLogger = setup_logger('dev_log', f'./logs/{self.args.dataset_name}/cross2/dev_results_{self.args.random_seed}.log')
			testLogger = setup_logger('test_log', f'./logs/{self.args.dataset_name}/cross2/test_results_{self.args.random_seed}.log')

		#Dump Validation Set Results
		devLogger.info(best_val_gold)
		devLogger.info(get_tuples(best_val_pred))
		devLogger.info('F1:{best_val_f1} on Epoch:{best_val_epoch}'.format(best_val_f1=best_val_f1, best_val_epoch=best_val_epoch))
		
		if self.args.crossType == 0:
			out_file = open(f'./logs/{self.args.dataset_name}/in-domain/dev_results_{self.args.random_seed}.json', "w")
		elif self.args.crossType == 1:
			out_file = open(f'./logs/{self.args.dataset_name}/cross1/dev_results_{self.args.random_seed}.json', "w")
		else:
			out_file = open(f'./logs/{self.args.dataset_name}/cross2/dev_results_{self.args.random_seed}.json', "w")

		valDict = dict()
		valDict['gold'] = best_val_gold 
		valDict['pred'] = get_tuples(best_val_pred) 
		json.dump(valDict, out_file, indent = 6)
		out_file.close()
		
		#Dump Test Set Results
		testLogger.info(best_saved_test_gold)
		testLogger.info(get_tuples(best_saved_test_pred))
		# testLogger.info('F1:{f1_test} on Epoch:{best_test_epoch}'.format(f1_test=f1_test, best_test_epoch=best_saved_test_epoch))
		testLogger.info('F1:{f1_test} on Epoch:{best_test_epoch}'.format(f1_test=best_saved_test_f1, best_test_epoch=best_val_epoch))

		if self.args.crossType == 0:
			out_file = open(f'./logs/{self.args.dataset_name}/in-domain/test_results_{self.args.random_seed}.json', "w")
		if self.args.crossType == 1:
			out_file = open(f'./logs/{self.args.dataset_name}/cross1/test_results_{self.args.random_seed}.json', "w")
		else:
			out_file = open(f'./logs/{self.args.dataset_name}/cross2/test_results_{self.args.random_seed}.json', "w")

		testDict = dict()
		testDict['gold'] = best_saved_test_gold 
		testDict['pred'] = get_tuples(best_saved_test_pred) 
		json.dump(testDict, out_file, indent = 6)
		out_file.close()

	def eval_model(self, eval_loader):
		self.model.eval()
		# print(self.model.decoder.query_embed.weight)
		prediction, gold = {}, {}
		with torch.no_grad():
			batch_size = self.args.batch_size
			eval_num = len(eval_loader)
			total_batch = eval_num // batch_size + 1
			for batch_id in range(total_batch):
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size
				if end > eval_num:
					end = eval_num
				eval_instance = eval_loader[start:end]
				if not eval_instance:
					continue
				input_ids, attention_mask, target, info = self.model.batchify(eval_instance)
				gold.update(formulate_gold(target, info))
				# print(target)
				gen_triples = self.model.gen_triples(input_ids, attention_mask, info)
				prediction.update(gen_triples)
		
		# num_metric(prediction, gold)
		overlap_metric(prediction, gold)
		return metric(prediction, gold), prediction, gold

	def load_state_dict(self, state_dict):
		self.model.load_state_dict(state_dict)

	@staticmethod
	def lr_decay(optimizer, epoch, decay_rate):
		# lr = init_lr * ((1 - decay_rate) ** epoch)
		if epoch != 0:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * (1 - decay_rate)
				# print(param_group['lr'])
		return optimizer
