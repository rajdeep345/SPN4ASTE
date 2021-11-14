import os
import json
import time
import pickle
import argparse
import warnings
from tqdm import tqdm
import multiprocessing
from functools import partial
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

#---------------------------------------------------------------------------------------------------

def get_f_pre_rec(gt_quad, pred_quad, correct_quad):
	test_p = float(correct_quad) / (pred_quad + 1e-8)
	test_r = float(correct_quad) / (gt_quad + 1e-8)
	test_f1 = (2 * test_p * test_r) / (test_p + test_r + 1e-8)
	return test_f1, test_p, test_r 


def get_splitted_F1(gt_triplets, pred_triplets):
	sent_count = 0
	gt_triplet = 0
	pred_triplet = 0
	correct_triplet = 0
	
	count_single = 0
	gt_single = 0
	pred_single = 0
	correct_single = 0
	
	count_multi = 0
	gt_multi = 0
	pred_multi = 0
	correct_multi = 0
	
	count_multiRel = 0
	gt_multiRel = 0
	pred_multiRel = 0
	correct_multiRel = 0
	
	count_overlappingEnt = 0
	gt_overlappingEnt = 0
	pred_overlappingEnt = 0
	correct_overlappingEnt = 0
	
	sent_count += 1
	correct_count = 0
	gt_triplet += len(gt_triplets)
	pred_triplet += len(pred_triplets)

	for p_triplet in pred_triplets:
		if p_triplet in gt_triplets:
			correct_count += 1
			correct_triplet += 1

	if len(gt_triplets) == 1:
		count_single += 1
		gt_single += len(gt_triplets)
		pred_single += len(pred_triplets)
		correct_single += correct_count  
	else:
		count_multi += 1
		gt_multi += len(gt_triplets)
		pred_multi += len(pred_triplets)
		correct_multi += correct_count

		rels = list()
		for g_triplet in gt_triplets:
			rels.append(g_triplet[0])	#rel at 0th index
		
		unique_rels = set(rels)
		if len(unique_rels) > 1:
			count_multiRel += 1
			gt_multiRel += len(gt_triplets)
			pred_multiRel += len(pred_triplets)
			correct_multiRel += correct_count
		
		flag = 0
		gt_triplets = list(gt_triplets)
		for j in range(len(gt_triplets)):
			for k in range(len(gt_triplets)):
				if j == k:
					continue
				if gt_triplets[j][1] >= gt_triplets[k][1] and gt_triplets[j][1] <= gt_triplets[k][2]:
					flag = 1
					break
				if gt_triplets[j][3] >= gt_triplets[k][3] and gt_triplets[j][3] <= gt_triplets[k][4]:
					flag = 1
					break
				if gt_triplets[j][1] >= gt_triplets[k][3] and gt_triplets[j][1] <= gt_triplets[k][4]:
					flag = 1
					break
				if gt_triplets[j][3] >= gt_triplets[k][1] and gt_triplets[j][3] <= gt_triplets[k][2]:
					flag = 1
					break                   
			if flag == 1:
				break
		if flag == 1:
			count_overlappingEnt += 1
			gt_overlappingEnt += len(gt_triplets)
			pred_overlappingEnt += len(pred_triplets)
			correct_overlappingEnt += correct_count
	
	return  sent_count, gt_triplet, pred_triplet, correct_triplet, count_single, gt_single, pred_single, correct_single, \
			count_multi, gt_multi, pred_multi, correct_multi, count_multiRel, gt_multiRel, pred_multiRel, correct_multiRel, \
			count_overlappingEnt, gt_overlappingEnt, pred_overlappingEnt, correct_overlappingEnt


def get_split_metrics(data, gt_tups, pred_tups):    
	counts=dict()
	counts['total_quad_count'] = 0
	counts['total_gt_quad'] = 0
	counts['total_pred_quad'] = 0
	counts['total_correct_quad'] = 0
	counts['total_count_single'] = 0
	counts['total_gt_single'] = 0
	counts['total_pred_single'] = 0
	counts['total_correct_single'] = 0
	counts['total_count_multi'] = 0
	counts['total_gt_multi'] = 0
	counts['total_pred_multi'] = 0
	counts['total_correct_multi'] = 0
	counts['total_count_multiRel'] = 0
	counts['total_gt_multiRel'] = 0
	counts['total_pred_multiRel'] = 0
	counts['total_correct_multiRel'] = 0
	counts['total_count_overlappingEnt'] = 0
	counts['total_gt_overlappingEnt'] = 0
	counts['total_pred_overlappingEnt'] = 0
	counts['total_correct_overlappingEnt'] = 0

	assert len(gt_tups) == len(pred_tups) == len(data)
	for i, sentid in enumerate(gt_tups):
		
		sent_count, gt_quad, pred_quad, correct_quad, count_single, gt_single, pred_single, correct_single, \
		count_multi, gt_multi, pred_multi, correct_multi, count_multiRel, gt_multiRel, pred_multiRel, correct_multiRel, \
		count_overlappingEnt, gt_overlappingEnt, pred_overlappingEnt, correct_overlappingEnt = get_splitted_F1(gt_tups[sentid],pred_tups[sentid])

		counts['total_gt_quad'] += gt_quad
		counts['total_pred_quad'] += pred_quad
		counts['total_correct_quad'] += correct_quad
		counts['total_count_single'] += count_single
		counts['total_gt_single'] += gt_single
		counts['total_pred_single'] += pred_single
		counts['total_correct_single'] += correct_single
		counts['total_count_multi'] += count_multi
		counts['total_gt_multi'] += gt_multi
		counts['total_pred_multi'] += pred_multi
		counts['total_correct_multi'] += correct_multi
		counts['total_count_multiRel'] += count_multiRel
		counts['total_gt_multiRel'] += gt_multiRel
		counts['total_pred_multiRel'] += pred_multiRel
		counts['total_correct_multiRel'] += correct_multiRel
		counts['total_count_overlappingEnt'] += count_overlappingEnt
		counts['total_gt_overlappingEnt'] += gt_overlappingEnt
		counts['total_pred_overlappingEnt'] += pred_overlappingEnt
		counts['total_correct_overlappingEnt'] += correct_overlappingEnt
		print(".", end=" ")

	return get_metric(counts)

def get_metric(counts):
	res2 = {}
	f, pre, rec = get_f_pre_rec(counts['total_gt_quad'], counts['total_pred_quad'], counts['total_correct_quad'])
	res2['quad_f'] = round(f, 4)*100
	res2['quad_rec'] = round(rec, 4)*100
	res2['quad_pre'] = round(pre, 4)*100

	f, pre, rec = get_f_pre_rec(counts['total_gt_single'], counts['total_pred_single'], counts['total_correct_single'])
	res2['single_f'] = round(f, 4)*100
	res2['single_rec'] = round(rec, 4)*100
	res2['single_pre'] = round(pre, 4)*100
	
	f, pre, rec = get_f_pre_rec(counts['total_gt_multi'], counts['total_pred_multi'], counts['total_correct_multi'])
	res2['multi_f'] = round(f, 4)*100
	res2['multi_rec'] = round(rec, 4)*100
	res2['multi_pre'] = round(pre, 4)*100

	f, pre, rec = get_f_pre_rec(counts['total_gt_multiRel'], counts['total_pred_multiRel'], counts['total_correct_multiRel'])
	res2['multiRel_f'] = round(f, 4)*100
	res2['multiRel_rec'] = round(rec, 4)*100
	res2['multiRel_pre'] = round(pre, 4)*100

	f, pre, rec = get_f_pre_rec(counts['total_gt_overlappingEnt'], counts['total_pred_overlappingEnt'], counts['total_correct_overlappingEnt'])
	res2['overlappingEnt_f'] = round(f, 4)*100
	res2['overlappingEnt_rec'] = round(rec, 4)*100
	res2['overlappingEnt_pre'] = round(pre, 4)*100

	return res2

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

if __name__ == "__main__":  
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_name', default='laptop', type=str)
	parser.add_argument('--crossType', default='in-domain', type=str)	
	parser.add_argument('--split', default='test', type=str)
	args= parser.parse_args()

	split = args.split
	dataset_name = args.dataset_name
	crossType = args.crossType
	results_file = f'logs/{dataset_name}/{crossType}/{split}_results_43.json'
	res_f = json.load(open(results_file))
	gt_tups = res_f['gold']
	pred_tups = res_f['pred']

	data_path = f'data/{crossType}/{dataset_name}_SPN_data.pickle'
	with open(data_path, 'rb') as fp:
	  file_data = pickle.load(fp)
	print("Data setting is loaded from file: ", data_path)
	file_data.show_data_summary()
	
	if split=='test':
		data = file_data.test_loader
	elif split=='dev':
		data = file_data.valid_loader

	res2 = get_split_metrics(data, gt_tups, pred_tups)   #args are at dataset level
	print("\n\nRESULTS FOR {dataset_name}:".format(dataset_name=dataset_name))
	print("\n\033[96mSINGLE-MULTI-CONTRAST-OVERLAP SPLIT\033[0m\n")
	print(res2)
	print("\n\033[96mOVERALL PERFORMANCE\033[0m")
	print("SPLIT2", res2['quad_pre']/100, res2['quad_rec']/100, res2['quad_f']/100) #Overall F1, P, R as calculated by 2nd splitting method
