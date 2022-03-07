import codecs
import pdb
import numpy as np
import nltk
import os 
import re
import random
import argparse
# from nltk.corpus import stopwords
from conlleval_perl import *



args=parse_args()
class Item:
	# Item Node: for one item (the key is the item string), it can have multiple queries
	def __init__(self,sequence):
		self.string = sequence
		self.tokens = nltk.word_tokenize(self.string)
		self.annotations = ['B-X'] * len(self.tokens)
		self.item_id = {}
		self.query = {}
		self.is_gold = False
	def add_query(self,query):
		# if query.string not in self.query:
		# 	self.query[query.string]=[]
		if query.string in self.query:
			return
		self.query[query.string]=query
	def add_item_id(self,item_id):
		if item_id not in self.item_id:
			self.item_id[item_id]=0
		self.item_id[item_id]+=1
	def __str__(self):
		return self.string
	def __getitem__(self, idx: int):
		return self.tokens[idx]
	def __iter__(self):
		return iter(self.tokens)
	def __repr__(self):
		return self.string
	def __len__(self) -> int:
		return len(self.tokens)

def check_span(item, just_return = False):
	current_ner=[]
	current_label='O'
	current_tag='O'
	span=0
	annos = item.annotations
	if just_return:
		annos = item.gold_annotations
	for i,tag in enumerate(annos):
		# pdb.set_trace()
		if tag=='B-X':
			continue
		# if tag == 'O' and current_label!='O':
		# 	current_ner.append([i,i+1,current_label])
		# 	current_label = tag
		# 	current_tag = tag
		# 	continue
		if '-' not in tag and current_label=='O':
			current_tag=tag
			current_label = tag
			continue
		elif '-' not in tag and current_label!='O':
			current_ner.append([i-span-1,i,current_label])
			current_label=tag
			current_tag=tag
			span=0
		else:
			label = tag.split('-')[-1]
			if 'I-' in tag and current_label==label:
				span+=1
				current_tag=tag
				current_label=label
			elif 'I-' in tag and current_label!=label:
				if (current_label == 'MEASUREMENT&PRODUCT' and (label == 'MEASUREMENT' or label == 'PRODUCT')) or (label == 'MEASUREMENT&PRODUCT' and (current_label == 'MEASUREMENT' or current_label == 'PRODUCT')) or (current_label == 'PER.NAM' and (label == 'PER.NOM')):
					span+=1
					current_tag=tag
					current_label=label
				elif current_label == 'O':
					current_tag=tag
					current_label=label
					span=0 
				else:
					current_ner.append([i-span-1,i,current_label])
					current_tag=tag
					current_label=label
					span=0 
					# pdb.set_trace()
					# print('Wrong data!')
					# print(item.string)
					# print(item.annotations)
					# item.ner_span = []
					# return
			elif 'B-' in tag:
				if current_tag!='O' or current_label!='O':
					current_ner.append([i-span-1,i,current_label])
				current_tag=tag
				current_label=label
				span=0  
			else:
				pdb.set_trace()
				print('Wrong data!')
				print(item.string)
				print(item.annotations)
				item.ner_span = []
				return
	if '-' in current_tag:
		current_ner.append([i-span,i+1,current_label])
	# print(current_ner)
	# print(ner_tags)
	if just_return:
		return current_ner
	item.ner_span = current_ner
	return

def count_file(file,target='train',is_file=False,max_len=100, comment_label = None, disable_print = False):
	if not is_file:
		filelist=os.listdir(file)
		for target_file in filelist:
			if 'swp' in target_file:
				continue
			if target in target_file:
				break
			# if 'train' in target_file:
			#   if 'train_new' in target_file:
			#       # os.remove(os.path.join(file,target_file))
			#       pass
			#   else:
			#       break
		# pdb.set_trace()
		#if write:
		to_write=os.path.join(file,target_file)
	else:
		target_file=file
		to_write=file
	reader=open(to_write,'r')
	# f=codecs.open(to_write,'r')
	# reader = codecs.getreader('utf-8')(f)
	# reader = codecs.getreader('latin1')(f)
	lines=reader.readlines()
	sentences=[]
	sentence=[]
	sent_length=[]
	for line in lines:
		line=line.strip()
		if comment_label is not None and line.startswith(comment_label):
			continue
		if line:
			sentence.append(line)
		elif sentence != []:
			sent_length.append(len(sentence))
			sentences.append(sentence.copy())
			sentence=[]
	if sentence != []:
		sent_length.append(len(sentence))
		sentences.append(sentence.copy())
		sentence=[]
	sent_length=np.array(sent_length)
	reader.close()
	if not disable_print:
		print(target_file, sent_length.max(),len(sentences),(sent_length>max_len).sum())
	# for sent in sentences:
	#   if len(sent)>100:
	#       print(sent)
	#       pdb.set_trace()
	return to_write,sentences
	# pdb.set_trace()

	
def write_file(to_write,sentences,max_len=100):
	write_file=to_write
	writer=open(write_file,'w')
	remove_count=0
	for sentence in sentences:
		if len(sentence)>max_len:
			remove_count+=1
			continue
		for word in sentence:
			writer.write(word+'\n')
		writer.write('\n')
	writer.close()
	print(f"Removed {remove_count} sentences that is longer than {max_len}")


def gen_origin_sentence(sentences, lang = None, dataset_name = '', window = 0):
	flatted_sentences = []
	for idx, sentence in enumerate(sentences):
		keyword = gen_sentence(sentence, lang = lang)
		item_idx = dataset_name+'_'+str(idx)
		flatted_sentences.append(lang+'\t'+item_idx+'\t'+keyword)
	return flatted_sentences


def replace_zh_space(text):
	match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
	should_replace_list = match_regex.findall(text)
	order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
	for i in order_replace_list:
		if i == u' ':
			continue
		new_i = i.strip()
		text = text.replace(i,new_i)
	return text


def gen_sentence(sentence, lang = None):
	keyword=' '.join([word.split()[0] for word in sentence])
	if lang is not None and (lang == 'zh' or lang =='mix'):
		keyword = replace_zh_space(keyword)
	return keyword

def gen_ner_based_query(sentences, lang = None, dataset_name = '', window = 0):
	all_items = []
	flatted_queries = []
	all_entities_count = 0
	append_count = 0 
	old_lang = lang
	for sent_id, sentence in enumerate(sentences):
		lang = old_lang
		tokens = [x.split()[0] for x in sentence]
		annos = [x.split()[-1] for x in sentence]
		sequence = gen_sentence(sentence,lang=lang)
		item = Item(sequence)
		item.annotations = annos
		item.tokens = tokens
		check_span(item)
		item.idx = dataset_name+'_'+str(sent_id)
		all_items.append(item)
		if lang == 'mix' and len(re.findall(r'[\u4e00-\u9fff]+', sequence))>0:
			lang = 'zh'
		all_entities_count+=len(item.ner_span)
		count_b = len([x for x in annos if 'B-' in x])
		if len(item.ner_span) != count_b:
			pdb.set_trace()
		for span in item.ner_span:
			# pdb.set_trace()
			if span[2] == 'O':
				pdb.set_trace()
				continue
			span_start = span[0]-window
			span_end = span[1]+window
			if span_start<0:
				span_start = 0
			if span_end > len(sentence):
				span_end = len(sentence)
			entity = tokens[span_start:span_end]
			flatted_queries.append(old_lang+'\t'+item.idx+'\t'+gen_sentence(entity,lang=lang)+'\t'+gen_sentence(sentence,lang=lang))
			append_count+=1
	print(f'Num Entities: {all_entities_count}')
	print(f'Num appends: {append_count}')
	print(len(flatted_queries))
	return flatted_queries


def gen_span_based_sentence(sentences, lang = None, dataset_name = '', window = 0):
	all_items = []
	flatted_queries = []
	all_entities_count = 0
	append_count = 0 
	old_lang = lang
	new_sentences = []
	for sent_id, sentence in enumerate(sentences):
		lang = old_lang
		tokens = [x.split() for x in sentence]
		new_sentence = []
		for token in tokens:
			if token[-1]!='O' and token[-1]!='B-X':
				token[-1] = token[-1].split('-')[0]+'-ENT'
			new_sentence.append(' '.join(token))
		new_sentences.append(new_sentence)
	return new_sentences


def gen_ner_based_query2(sentence, lang = None, dataset_name = '', window = 0):
	all_items = []
	all_entities_count = 0
	append_count = 0 
	flatted_queries = []
	old_lang = lang
	lang = old_lang
	tokens = [x.split()[0] for x in sentence]
	annos = [x.split()[-1] for x in sentence]
	sequence = gen_sentence(sentence,lang=lang)
	item = Item(sequence)
	item.annotations = annos
	item.tokens = tokens
	check_span(item)
	all_items.append(item)
	if lang == 'mix' and len(re.findall(r'[\u4e00-\u9fff]+', sequence))>0:
		lang = 'zh'
	all_entities_count+=len(item.ner_span)
	count_b = len([x for x in annos if 'B-' in x])
	if len(item.ner_span) != count_b:
		pdb.set_trace()
	for span in item.ner_span:
		# pdb.set_trace()
		if span[2] == 'O':
			pdb.set_trace()
			continue
		span_start = span[0]-window
		span_end = span[1]+window
		if span_start<0:
			span_start = 0
		if span_end > len(sentence):
			span_end = len(sentence)
		entity = tokens[span_start:span_end]
		flatted_queries.append(gen_sentence(entity,lang=lang))
		append_count+=1
	return flatted_queries

def convert_json_to_conll(sentences):
	new_sentences= []
	for sentence in sentences:
		tokens = [word+'\tO' for word in sentence['tokens']]
		for entity in sentence['entities']:
			for i in range(entity['start'],entity['end']):
				cols = tokens[i].split('\t')
				if i == entity['start']:
					cols[-1] = 'B-'+entity['type']
				else:
					cols[-1] = 'I-'+entity['type']
				tokens[i]='\t'.join(cols)
		new_sentences.append(tokens)
	return new_sentences


def bioes2bio(annos):
	for idx, anno in enumerate(annos):
		if anno.startswith('S-'):
			annos[idx] = re.sub('S-', 'B-', anno)
		if anno.startswith('E-'):
			annos[idx] = re.sub('E-', 'I-', anno)
	return annos


def label2mention(annos):
	for idx, anno in enumerate(annos):
		if anno.startswith('B-'):
			annos[idx] = 'B-ENT'
		if anno.startswith('I-'):
			annos[idx] = 'I-ENT'
	return annos


def gen_item(sentences, lang = None, dataset_name = '', vote_dict = None, ignore_label=False, num_column = 4):
	all_items = []
	flatted_queries = []
	all_entities_count = 0
	append_count = 0 
	old_lang = lang
	for sent_id, sentence in enumerate(sentences):
		lang = old_lang
		tokens = []
		gold_annos = []
		pred_annos = []
		for x in sentence:
			if num_column == 4:
				try:
					token, gold_anno, pred_anno, _ = x.split()
				except:
					pdb.set_trace()
			elif num_column == 3:
				token, gold_anno, pred_anno = x.split()
			if token == '<EOS>' or gold_anno == 'S-X':
				break
			tokens.append(token)
			gold_annos.append(gold_anno)
			pred_annos.append(pred_anno)
		gold_annos = bioes2bio(gold_annos)
		pred_annos = bioes2bio(pred_annos)
		if ignore_label:
			gold_annos = label2mention(gold_annos)
			pred_annos = label2mention(pred_annos)
		sequence = gen_sentence(tokens,lang=lang)
		item = Item(sequence)
		item.annotations = pred_annos
		item.gold_annotations = gold_annos
		item.tokens = tokens
		check_span(item)
		if vote_dict is None:
			continue
		if sequence not in vote_dict:
			vote_dict[sequence] = {}
		for span in item.ner_span:
			# pdb.set_trace()
			if span[2] == 'O':
				pdb.set_trace()
				continue
			span = tuple(span)
			if span not in vote_dict[sequence]:
				vote_dict[sequence][span] = 0			
			vote_dict[sequence][span]+=1
		flatted_queries.append(item)
	return flatted_queries, vote_dict

def pred_ensemble(vote_dict, items, vote_num = -1, threshold = 0):
	for item in items:
		vote = vote_dict[item.string]
		new_annos = ['O' for i in item.annotations]
		# vote={(7,10,'CORP'):6, (7,10,'ABC'):7, (7,11,'ABC'):7, (7,11,'CORP'):6, (7,11,'CW'):6, (7,15,'CORP'):5}
		sorted_list = sorted(vote.items(), key=lambda item: item[0][1]-item[0][0], reverse=True)
		sorted_vote = sorted(sorted_list, key=lambda item: item[1], reverse=True)
		# pdb.set_trace()
		for span, value in sorted_vote:
			start, end, label = span
			if value < threshold:
				continue
			if set(new_annos[start:end]) == set(['O']):
				new_annos[start] = 'B-'+label
				for i in range(start+1,end):
					new_annos[i] = 'I-'+label
		item.ensemble_annos = new_annos
	return items


def pred_recall(vote_dict, items, vote_num = -1, threshold = 0):
	tp = 0
	tn = 0
	fp = 0
	for item in items:
		vote = vote_dict[item.string]
		new_annos = ['O' for i in item.annotations]
		# vote={(7,10,'CORP'):6, (7,10,'ABC'):7, (7,11,'ABC'):7, (7,11,'CORP'):6, (7,11,'CW'):6, (7,15,'CORP'):5}
		sorted_list = sorted(vote.items(), key=lambda item: item[0][1]-item[0][0], reverse=True)
		sorted_vote = sorted(sorted_list, key=lambda item: item[1], reverse=True)
		gold_spans = check_span(item, just_return = True)
		# pdb.set_trace()
		gold_spans = set([(s, e) for s, e, l in gold_spans])
		pred_spans = set([(s, e) for (s, e, l), num in sorted_vote if num>threshold])
		cross = len(gold_spans & pred_spans)
		tp += cross
		fp += len(pred_spans) - cross
		tn += len(gold_spans) - cross
	print(f'precision: {tp/(tp+fp) * 100}',f'recall: {tp/(tp+tn) * 100}')
	return items
				
def pred_spans(vote_dict, items, vote_num = -1, threshold = 0, lang = None):
	tp = 0
	tn = 0
	fp = 0
	lines = []
	for item in items:
		vote = vote_dict[item.string]
		new_annos = ['O' for i in item.annotations]
		# vote={(7,10,'CORP'):6, (7,10,'ABC'):7, (7,11,'ABC'):7, (7,11,'CORP'):6, (7,11,'CW'):6, (7,15,'CORP'):5}
		sorted_list = sorted(vote.items(), key=lambda item: item[0][1]-item[0][0], reverse=True)
		sorted_vote = sorted(sorted_list, key=lambda item: item[1], reverse=True)
		pred_spans = set([(s, e) for (s, e, l), num in sorted_vote if num>threshold])
		sequence = gen_sentence(item.tokens,lang=lang)
		span_tokens = []
		span_tokens.append(sequence)
		for span in pred_spans:
			span_tokens.append(gen_sentence(item.tokens[span[0]:span[1]], lang = lang))
		# pdb.set_trace()
		lines.append('\t'.join(span_tokens))
	return lines

if __name__ == '__main__':
	sent_input_file_groups11= [
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner20.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner21...conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner22.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner23.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner24.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner25.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner26.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner28.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner29.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner30.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner31.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner32.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner33.v1..conllu',
	'en_ensemble_example/train.xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner34.v1..conllu',
	]

	# langs=['tr', 'de', 'fa', 'ko', 'zh', 'es', 'mix', 'hi', 'ru', 'nl', 'bn', 'en', 'multi']
	langs = ['en']
	# mentions = [mention_groups1, mention_groups2, mention_groups3, mention_groups4, mention_groups5, mention_groups6, mention_groups7, mention_groups8, mention_groups9, mention_groups10, mention_groups11, mention_groups12]

	# entities = [input_file_groups1, input_file_groups2, input_file_groups3, input_file_groups4, input_file_groups5, input_file_groups6, input_file_groups7, input_file_groups8, input_file_groups9, input_file_groups10, input_file_groups11, input_file_groups12]
	sentences = [sent_input_file_groups11]
	# sentences = [sent_input_file_groups0, sent_input_file_groups1, sent_input_file_groups2, sent_input_file_groups3, sent_input_file_groups4, sent_input_file_groups5, sent_input_file_groups6, sent_input_file_groups7, sent_input_file_groups8, sent_input_file_groups9, sent_input_file_groups10, sent_input_file_groups11, sent_input_file_groups12]

	# multi1 = [sent_input_file_groups_multi, mention_groups_multi, input_file_groups_multi]
	# multi2 = [sent_input_file_groups_multi2, mention_groups_multi2, input_file_groups_multi2]

	threshold = 0.5
	add_multi = False
	for group_id, big_group in enumerate([sentences]):
		if group_id != 0: 
			continue
		for small_group_id, group in enumerate(big_group):
			input_file_groups = group
			all_pred_sentences = []
			vote_dict = {}
			for input_file in input_file_groups:
				to_write,pred_sentences=count_file(input_file,'train',is_file=True, comment_label = '# id', disable_print = False)
				flatted_queries, vote_dict = gen_item(pred_sentences, vote_dict = vote_dict, ignore_label = False)
				all_pred_sentences.append(flatted_queries)


			items = pred_ensemble(vote_dict, all_pred_sentences[0], vote_num = len(input_file_groups), threshold = len(input_file_groups) * threshold)

			# items = pred_recall(vote_dict, all_pred_sentences[0], vote_num = len(input_file_groups))
			# # continue
			# lang = langs[small_group_id]
			# lines = pred_spans(vote_dict, all_pred_sentences[0], lang = lang)

			evaluate_sentences = []
			new_sentences = []
			for item in items:
				sentence = []
				for idx, token in enumerate(item.tokens):
					sentence.append(' '.join([token, item.gold_annotations[idx], item.ensemble_annos[idx]]))
				new_sentences.append(sentence)
				evaluate_sentences+=sentence
				evaluate_sentences+=['\n']
			lang = langs[small_group_id]
			name = 'semeval2022_ensemble/'+lang + '.ensem_recal_pred_'+args.name+'.conll'
			writer = open(name, 'w')
			print(name)
			for line in evaluate_sentences:
				writer.write(line+'\n')
			continue
			correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter = countChunks(evaluate_sentences, args)
			# compute metrics and print
			print(langs[small_group_id], threshold)
			evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter, latex=args.latex)


		# write_file('pred.tsv', new_sentences, max_len = 999)
