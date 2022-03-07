import numpy as np
import time
import torch
import torch.nn as nn
# from flair.parser.modules.dropout import SharedDropout

from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
								pad_sequence)

import flair.ner_dp_utils as utils
from flair.ner_dp_utils import get_shape, train_dataloader, eval_dataloader


import pdb


class BiaffineNERModel(nn.Module):
	def __init__(self, config, model_sizes):
		super().__init__()
		self.config = config
		self.device = torch.device(config['device'])
		# self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
		
		
		self.char_dict = utils.load_char_dict(config["char_vocab_path"])
		self.char_emb_size = config["char_embedding_size"]
		self.char_filter_widths = config['filter_widths']
		self.char_filter_size = config['filter_size']
		self.char_wordemb_size = len(self.char_filter_widths)*self.char_filter_size
		self.char_embbedings = torch.nn.Embedding(num_embeddings=len(self.char_dict), embedding_dim=self.char_emb_size)

		self.context_embeddings_size = model_sizes[0] + self.char_wordemb_size

		self.eval_data = None  # Load eval data lazily.
		self.ner_types = self.config['ner_types']
		self.ner_maps = {ner: (i + 1) for i, ner in enumerate(self.ner_types)}
		self.num_types = len(self.ner_types)

		self.dropout = self.config["dropout_rate"]
		# self.lexical_Dropout = torch.nn.Dropout(p=config['lexical_dropout_rate'])
		self.lstm_dropout = self.config["lstm_dropout_rate"]
		self.lexical_Dropout = nn.Dropout(p=config['lexical_dropout_rate'])

		self.lstm_input_size = self.context_embeddings_size
		self.lstm_output_size = 2*self.config["contextualization_size"]
		self.mlpx = projection(emb_size=self.context_embeddings_size, 
								output_size=self.lstm_output_size)
		#char emb
		self.char_emb_cnn = cnn(emb_size=self.char_emb_size, kernel_sizes=self.char_filter_widths, num_filter=self.char_filter_size)
		
		
		self.rnn = BiLSTM_1(input_size=self.lstm_input_size,
							hidden_size=self.config["contextualization_size"],
							num_layers=config['contextualization_layers'],
							dropout=self.lstm_dropout)
		
		self.start_project = projection(emb_size=self.lstm_output_size,
										output_size=self.config["ffnn_size"])
		self.end_project = projection(emb_size=self.lstm_output_size,
										output_size=self.config['ffnn_size'])

		self.bilinear = bilinear_classifier(dropout=self.dropout, 
											input_size_x=self.config["ffnn_size"], 
											input_size_y=self.config["ffnn_size"],
											output_size=self.num_types+1,
											)
		self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
		self.global_step = 0
		self.batch_len = None

		self.to(self.device)
	
	def sequence_mask(self, lengths, maxlen, dtype=torch.bool):
		if maxlen is None:
			maxlen = lengths.max()
		row = torch.range(0, maxlen-1).to(self.device )
		matrix = torch.tensor(lengths).view(-1,1)
		mask = matrix > row
		mask.type(dtype)
		return mask

	def forward(self, batch, is_train=False):
		"""compute score for each step"""
		batch_tensors = batch[0]
		tokens, context_word_emb, char_index, text_len, gold_labels = batch_tensors

		n_sentences, max_sentence_length = tokens.shape[0], tokens.shape[1]
		text_len_mask = self.sequence_mask(lengths=text_len, maxlen=max_sentence_length)

		context_emb_list = []
		context_emb_list.append(context_word_emb)

		#TODO add char_emb
		# pdb.set_trace()
		char_emb = self.char_embbedings(torch.as_tensor(char_index, device=self.device, dtype=torch.int64))
		_, _, max_char_len, self.char_emb_size = char_emb.shape
		flattened_char_emb = char_emb.reshape([n_sentences * max_sentence_length, max_char_len, self.char_emb_size]).transpose_(1,2)	# n_words, max_word_len, char_emb_size (N, L, C)->(N, C, L)
		flattened_aggregated_char_emb = self.char_emb_cnn(flattened_char_emb)
		aggregated_char_emb = flattened_aggregated_char_emb.reshape(n_sentences, max_sentence_length, flattened_aggregated_char_emb.shape[1])
		context_emb_list.append(aggregated_char_emb)
		# pdb.set_trace()
		context_emb = torch.cat(context_emb_list, 2)
		context_emb = self.lexical_Dropout(context_emb)

		candidate_scores_mask = torch.logical_and(torch.unsqueeze(text_len_mask,dim=1),torch.unsqueeze(text_len_mask,dim=2)) 
		candidate_scores_mask = torch.triu(candidate_scores_mask, diagonal=0)
		flattened_candidate_scores_mask = candidate_scores_mask.view(-1)

		# pdb.set_trace()
		#----------through rnn------------
		pack = pack_padded_sequence(context_emb, text_len, batch_first=True, enforce_sorted=False)
		pack, _ = self.rnn(pack)
		context_outputs, _ = pad_packed_sequence(pack, batch_first=True, total_length=context_emb.shape[1])

		# context_outputs = self.mlpx(context_emb)
		#--------biaffine----------------
		candidate_starts_emb = self.start_project(context_outputs)
		candidate_end_emb = self.end_project(context_outputs)

		
		candidate_ner_scores = self.bilinear(candidate_starts_emb, candidate_end_emb)
		candidate_ner_scores = candidate_ner_scores.reshape(-1,self.num_types+1)[flattened_candidate_scores_mask==True]
		# pdb.set_trace()
		if is_train:
			loss = self.criterion(input=candidate_ner_scores, target=gold_labels)
			loss = loss.sum()
		else:
			loss = 0
	
		return loss, candidate_ner_scores




	



	def get_pred_ner(self, sentences, span_scores, is_flat_ner):  # span_scores: shape [num_sentence, max_sentence_length,max_sentence_length,types+1]
		candidates = []
		span_scores = span_scores.detach().cpu().numpy()
		for sid,sent in enumerate(sentences):
			for s in range(len(sent)):
				for e in range(s,len(sent)):
					candidates.append((sid,s,e))
		
		top_spans = [[] for _ in range(len(sentences))]
		for i, type in enumerate(np.argmax(span_scores,axis=1)):  # span_scores (429,8), type: ner label index, i: sentence id.
			if type > 0:
				sid, s,e = candidates[i]
				top_spans[sid].append((s,e,type,span_scores[i,type]))

		top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans] # 对于每个句子中的所有spans，按分数排序
		sent_pred_mentions = [[] for _ in range(len(sentences))]
		for sid, top_span in enumerate(top_spans):
			for ns,ne,t,_ in top_span:
				for ts,te,_ in sent_pred_mentions[sid]:   # ts, te 是之前记录的span start/end position，ns, ne必须与所有的 ts,te相容。
					if ns < ts <= ne < te or ts < ns <= te < ne:      # clash 发生，break, ns, ne不相容，看下一span.
						#for both nested and flat ner no clash is allowed
						break
					if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
						#for flat ner nested mentions are not allowed
						break
				else:
					sent_pred_mentions[sid].append((ns,ne,t))
		# pdb.set_trace()
		pred_mentions = set((sid,s,e,t) for sid, spr in enumerate(sent_pred_mentions) for s,e,t in spr)
		return pred_mentions



	#------------------------------------------------------------
	# for training
	# def load_datasets(self, datatype='train'):
	# 	if datatype=='train':
	# 		self.train_dataloader = train_dataloader(config)
	# 		self.batch_len = len(self.train_dataloader)
	# 	elif datatype=='eval':
	# 		self.eval_dataloader = eval_dataloader(config)
	# 	else:
	# 		pdb.set_trace()
		
	# def step(self):
	# 	if self.batch_len is not None:
	# 		batch = self.train_dataloader[self.global_step%self.batch_len]
	# 		loss_step = self.forward(batch)
	# 	else:
	# 		pdb.set_trace()
	# 	self.global_step += 1




	def evaluate(self, eval_dataloader, is_final_test=False):
		# self.load_eval_data()
		self.eval()

		tp,fn,fp = 0,0,0
		start_time = time.time()
		num_words = 0
		sub_tp,sub_fn,sub_fp = [0] * self.num_types,[0]*self.num_types, [0]*self.num_types

		is_flat_ner = 'flat_ner' in self.config and self.config['flat_ner']

		for batch_num, batch in enumerate(eval_dataloader.batches):
			
			batch_tensor, batch_data = batch
			# tokens, context_word_emb, char_index, text_len, gold_labels = batch_tensor

			# pdb.set_trace()
			_, candidate_ner_scores = self.forward(batch, is_train=False) # (439, 8)

			num_words += sum(len(tok) for tok in batch_data['sentences'])


			gold_ners = set([(sid,s,e, self.ner_maps[t]) for sid, ner in enumerate(batch_data['ners']) for s,e,t in ner])  # {(1, 3, 3, 4), (0, 0, 0, 4), (1, 10, 14, 6), (0, 3, 3, 3), (1, 20, 25, 3), (0, 5, 5, 3), (1, 20, 20, 3), (1, 22, 22, 3)}
			pred_ners = self.get_pred_ner(batch_data["sentences"], candidate_ner_scores,is_flat_ner)

			tp += len(gold_ners & pred_ners)
			fn += len(gold_ners - pred_ners)
			fp += len(pred_ners - gold_ners)

			if is_final_test:
				for i in range(self.num_types):
					sub_gm = set((sid,s,e) for sid,s,e,t in gold_ners if t ==i+1)
					sub_pm = set((sid,s,e) for sid,s,e,t in pred_ners if t == i+1)
					sub_tp[i] += len(sub_gm & sub_pm)
					sub_fn[i] += len(sub_gm - sub_pm)
					sub_fp[i] += len(sub_pm - sub_gm)

			if batch_num % 10 == 0:
				print("Evaluated {}/{} examples.".format(batch_num + 1, len(eval_dataloader.batches)))

		used_time = time.time() - start_time
		print("Time used: %d second, %.2f w/s " % (used_time, num_words*1.0/used_time))

		m_r = 0 if tp == 0 else float(tp)/(tp+fn)
		m_p = 0 if tp == 0 else float(tp)/(tp+fp)
		m_f1 = 0 if m_p == 0 else 2.0*m_r*m_p/(m_r+m_p)

		print("Mention F1: {:.2f}%".format(m_f1*100))
		print("Mention recall: {:.2f}%".format(m_r*100))
		print("Mention precision: {:.2f}%".format(m_p*100))

		if is_final_test:
			print("****************SUB NER TYPES********************")
			for i in range(self.num_types):
				sub_r = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fn[i])
				sub_p = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fp[i])
				sub_f1 = 0 if sub_p == 0 else 2.0 * sub_r * sub_p / (sub_r + sub_p)

				print("{} F1: {:.2f}%".format(self.ner_types[i],sub_f1 * 100))
				print("{} recall: {:.2f}%".format(self.ner_types[i],sub_r * 100))
				print("{} precision: {:.2f}%".format(self.ner_types[i],sub_p * 100))

		summary_dict = {}
		summary_dict["Mention F1"] = m_f1
		summary_dict["Mention recall"] = m_r
		summary_dict["Mention precision"] = m_p

		return utils.make_summary(summary_dict), m_f1




		





#-----------modules------------------










#-----------bilinear-----------------

class Sparse_dropout(nn.Module):
	def __init__(self, p):
		super(Sparse_dropout, self).__init__()
		self.dropout_rate = p
	
	def forward(self, input, noise_shape):
		if not self.training:
			return input
		shapes = input.shape
		noise_shape = list(noise_shape)
		broadcast_dims = []
		# pdb.set_trace()
		for idx, dim_pair in enumerate(zip(shapes, noise_shape)):
			if dim_pair[1]>1:
				broadcast_dims.append((idx, dim_pair[0]))

		mask_dims = []
		for dim in broadcast_dims:
			mask_dims.append(dim[1])
		mask = torch.bernoulli((torch.ones(mask_dims, device=input.device)*(1-self.dropout_rate)).reshape(noise_shape))*(1/(1-self.dropout_rate))
		mask.to(input.dtype)
		return input*mask




class bilinear_classifier(nn.Module):

	def __init__(self, dropout, input_size_x, input_size_y, output_size, bias_x=True, bias_y=True):
		super(bilinear_classifier, self).__init__()
		
		# self.batch_size = batch_size
		# self.bucket_size = bucket_size
		# self.input_size = input_size
		# pdb.set_trace()
		# self.dropout_rate = 0
		self.dropout_rate = dropout
		self.output_size = output_size
		
		self.dropout = Sparse_dropout(p=self.dropout_rate)
		self.biaffine = biaffine_mapping(
						input_size_x, input_size_y,
						output_size, bias_x, bias_y,
						)
	def forward(self, x_bnv, y_bnv):
		batch_size, input_size_x = x_bnv.shape[0], x_bnv.shape[-1]
		input_size_y = y_bnv.shape[-1]
		noise_shape_x = [batch_size, 1, input_size_x]
		noise_shape_y = [batch_size, 1, input_size_y]
		x = self.dropout(x_bnv, noise_shape_x)
		y = self.dropout(y_bnv, noise_shape_y)

		output = self.biaffine(x, y)
		#TODO reshape output
		if self.output_size == 1:
		  output = output.squeeze(-1)
		return output

class biaffine_mapping(nn.Module):
	def __init__(self, input_size_x, input_size_y, output_size, bias_x, bias_y, initializer=None):
		super(biaffine_mapping, self).__init__()
		self.bias_x = bias_x
		self.bias_y = bias_y
		self.output_size = output_size
		self.initilizer = None
		if self.bias_x:
		  input_size1 = input_size_x + 1
		  input_size2 = input_size_y + 1
		self.biaffine_map = nn.Parameter(torch.Tensor(input_size1, output_size, input_size2))
		
		self.initialize()

	def initialize(self):
		if self.initilizer == None:
			torch.nn.init.orthogonal_(self.biaffine_map)
		else:
			self.initilizer(self.biaffine_map)


	def forward(self, x, y):
		batch_size, bucket_size = x.shape[0], x.shape[1]
		if self.bias_x:
		  x = torch.cat([x, torch.ones([batch_size, bucket_size, 1], device=x.device)], axis=2)
		if self.bias_y:
		  y = torch.cat([y, torch.ones([batch_size, bucket_size, 1], device=y.device)], axis=2)

		#reshape
		x_set_size, y_set_size = x.shape[-1], y.shape[-1]
		# b,n,v1 -> b*n, v1
		x = x.reshape(-1, x_set_size)
		# # b,n,v2 -> b*n, v2
		# y = y.reshape(-1, y_set_size)
		biaffine_map = self.biaffine_map.reshape(x_set_size, -1)  # v1, r, v2 -> v1, r*v2
		# b, n, r*v2 -> b, n*r, v2
		biaffine_mapping = (torch.matmul(x, biaffine_map)).reshape(batch_size, -1, y_set_size)
		# (b, n*r, v2) bmm (b, n, v2) -> (b, n*r, n) -> (b, n, r, n)
		biaffine_mapping = (biaffine_mapping.bmm(torch.transpose(y, 1, 2))).reshape(batch_size, bucket_size, self.output_size, bucket_size)
		# (b, n, r, n) -> (b, n, n, r)
		biaffine_mapping = biaffine_mapping.transpose(2, 3)

		return biaffine_mapping


#------------ linear --------------------------------------


def projection(emb_size, output_size, initializer=None):
  return ffnn(emb_size, 0, -1, output_size, dropout=0, output_weights_initializer=initializer)

class ffnn(nn.Module):
  
	def __init__(self, emb_size, num_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
		super(ffnn, self).__init__()
		
		self.dropout = torch.nn.Dropout(p=dropout)
		self.weights = nn.Parameter(torch.Tensor(emb_size, output_size))
		self.bias = nn.Parameter(torch.Tensor(output_size))
		self.activation = torch.nn.ReLU()
		self.num_layers = num_layers
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.initializer = output_weights_initializer

		self.initialize()
		
	def initialize(self):
		if self.initializer == None:
			torch.nn.init.xavier_uniform_(self.weights, gain=1)
		else:
			# pdb.set_trace()
			self.initializer(self.weights, gain=1)
		nn.init.zeros_(self.bias)

	def forward(self, inputs):
		# pdb.set_trace()
		current_inputs = inputs
		if len(get_shape(inputs))==3:
			batch_size, seqlen, emb_size = get_shape(inputs)
			current_inputs = inputs.reshape(batch_size*seqlen, emb_size)
		emb_size = get_shape(current_inputs)[-1]
		# if emb_size != self.emb_size:
		# 	pdb.set_trace()
		assert emb_size==self.emb_size,'last dim of input does not match this layer'
		
		# if self.dropout is not None or self.dropout > 0:
		# 	output = self.dropout(current_inputs)
		#TODO num_layers>0 case.

		outputs = current_inputs.matmul(self.weights) + self.bias

		if len(get_shape(inputs))==3:
			outputs = outputs.reshape(batch_size, seqlen, self.output_size)
		
		return outputs


#--------------lstm ---------------------

class BiLSTM_1(nn.Module):

	def __init__(self, input_size, hidden_size, num_layers, dropout=None):
		super(BiLSTM_1, self).__init__()

		self.input_size = input_size	#emb_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout_rate = dropout
		
		self.f_cells = nn.ModuleList()
		self.b_cells = nn.ModuleList()
		
		for _ in range(self.num_layers):
			self.f_cells.append(LstmCell(input_size, hidden_size, dropout))
			self.b_cells.append(LstmCell(input_size, hidden_size, dropout))

			input_size = 2*hidden_size

		self.dropout = torch.nn.Dropout(p=dropout)
		self.mlp = projection(emb_size=input_size, output_size=input_size)
		# self.initialize()

	def __repr__(self):
		s = self.__class__.__name__ + '('
		s += f"{self.input_size}, {self.hidden_size}"
		if self.num_layers > 1:
			s += f", num_layers={self.num_layers}"
		if self.dropout_rate > 0:
			s += f", dropout={self.dropout_rate}"
		s += ')'
		return s

	def permute_hidden(self, hx, permutation):
		if permutation is None:
			return hx
		h = apply_permutation(hx[0], permutation)
		c = apply_permutation(hx[1], permutation)

		return h, c

	def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
		hx_0 = hx_i = hx
		hx_n, output = [], []
		steps = reversed(range(len(x))) if reverse else range(len(x))
		# if self.training:
		#     hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

		for t in steps:
			last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
			if last_batch_size < batch_size:
				hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
						for h, ih in zip(hx_i, hx_0)]
			else:
				hx_n.append([h[batch_size:] for h in hx_i])
				hx_i = [h[:batch_size] for h in hx_i]
			# pdb.set_trace()
			hx_i = [h for h in cell(x[t], hx_i)]
			output.append(hx_i[0])
			# if self.training:
			#     hx_i[0] = hx_i[0] * hid_mask[:batch_size]
		if reverse:
			hx_n = hx_i
			output.reverse()
		else:
			hx_n.append(hx_i)
			hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
		# pdb.set_trace()
		output = torch.cat(output)

		return output, hx_n


	def forward(self, sequence, hx=None):
		# pdb.set_trace()
		x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
		
		batch_size = batch_sizes[0]
		h_n, c_n = [], []

		if hx is None:
			# pdb.set_trace()
			
			h = self.f_cells[0].initial_state[0].repeat([batch_size, 1])
			c = self.f_cells[0].initial_state[1].repeat([batch_size, 1])

			h = torch.unsqueeze(torch.unsqueeze(h, 0), 0).repeat([self.num_layers, 2, 1, 1])
			c = torch.unsqueeze(torch.unsqueeze(c, 0), 0).repeat([self.num_layers, 2, 1, 1])
		else:
			h, c = self.permute_hidden(hx, sequence.sorted_indices)
		h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
		c = c.view(self.num_layers, 2, batch_size, self.hidden_size)
		
		for i in range(self.num_layers):
			current_input = x
			x = torch.split(x, batch_sizes)
			
			# if self.training:
			# 	mask = SharedDropout.get_mask(x[0], self.dropout)
			# 	x = [i * mask[:len(i)] for i in x]
			x_f, (h_f, c_f) = self.layer_forward(x=x,
												 hx=(h[i,0], c[i,0]),
												 cell=self.f_cells[i],
												 batch_sizes=batch_sizes												 
												 )
			x_b, (h_b, c_b) = self.layer_forward(x=x,
												 hx=(h[i, 1], c[i, 1]),
												 cell=self.b_cells[i],
												 batch_sizes=batch_sizes,
												 reverse=True)			
			h_n.append(torch.stack((h_f, h_b)))
			c_n.append(torch.stack((c_f, c_b)))
			text_outputs = torch.cat((x_f, x_b), -1)
			text_outputs = self.dropout(text_outputs)

			if i > 0:
				# pdb.set_trace()
				highway_gates = torch.sigmoid(self.mlp(text_outputs))
				text_outputs = highway_gates*text_outputs + (1-highway_gates)*current_input
			x = text_outputs


		x = PackedSequence(x,
						   sequence.batch_sizes,
						   sequence.sorted_indices,
						   sequence.unsorted_indices)
		hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
		hx = self.permute_hidden(hx, sequence.unsorted_indices)

		return x, hx


	



class LstmCell(nn.Module):
	def __init__(self, input_size, hidden_size, dropout=0):
		super(LstmCell, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = torch.nn.Dropout(p=dropout)
		self.mlp = projection(emb_size=input_size+hidden_size, output_size=3*hidden_size, 
							initializer=self._block_orthonormal_initializer(output_sizes=[hidden_size] * 3)
							)
		
		self.initial_cell_state = nn.Parameter(torch.Tensor(1, hidden_size))
		self.initial_hidden_state = nn.Parameter(torch.Tensor(1, hidden_size))
		self.initialize()
		self._initial_state = (self.initial_cell_state, self.initial_hidden_state)
		

	def initialize(self):
		torch.nn.init.xavier_uniform_(self.initial_cell_state, gain=1)
		torch.nn.init.xavier_uniform_(self.initial_hidden_state, gain=1)

	def forward(self, inputs, states):
		batch_size = get_shape(inputs)[0]
		_dropout_mask = self.dropout(torch.ones(batch_size, self.hidden_size, device=inputs.device))
		h, c = states

		
		if self.training:
			h *= _dropout_mask
		concat = self.mlp(inputs=torch.cat([inputs, h], axis=1))	
		i, j, o = torch.chunk(input=concat, chunks=3, dim=1)
		i = torch.sigmoid(i)
		new_c = (1-i)*c + i*torch.tanh(j)		
		new_h = torch.tanh(new_c) * torch.sigmoid(o)	
		new_state = (new_h, new_c)
		return new_state

	@property
	def initial_state(self):
		return self._initial_state
	

	def _orthonormal_initializer(self, weights, gain=1.0):
		if len(weights.shape)>2:
			pdb.set_trace()
		device = weights.device
		dtype = weights.dtype
		# pdb.set_trace()
		shape0, shape1 = get_shape(weights)
		M1 = torch.randn(size=(shape0, shape0), dtype=dtype, device=device)
		M2 = torch.randn(size=(shape1, shape1), dtype=dtype, device=device)
		Q1, R1 = torch.qr(M1)	# let weights.shape= (s0,s1) and sm = min(s0, s1), then Q1:(s0,sm), R1:(sm,s1)
		Q2, R2 = torch.qr(M2)
		Q1 = Q1 * torch.sign(torch.diag(R1))
		Q2 = Q2 * torch.sign(torch.diag(R2))
		n_min = min(shape0, shape1)
		
		with torch.no_grad():
			q = torch.matmul(Q1[:, :n_min], Q2[:n_min, :])
			weights.view_as(q).copy_(q)
			weights.mul_(gain)
		return weights

	def _block_orthonormal_initializer(self, output_sizes):
		def _initializer(weights, gain=1.0):
			shape = get_shape(weights)
			assert len(shape) == 2
			assert sum(output_sizes) == shape[1]
			initializer = self._orthonormal_initializer
			
			
			with torch.no_grad():
				# pdb.set_trace()
				q_list = [initializer(a, gain) for a in torch.split(weights,split_size_or_sections=output_sizes, dim=1)]
				q = torch.cat(q_list, axis=1)
				weights.view_as(q).copy_(q)
			return weights
		return _initializer



#---------character embedding-----------------------

class cnn(nn.Module):
	def __init__(self, emb_size, kernel_sizes, num_filter):
		super(cnn, self).__init__()
		self.emb_size = emb_size
		self.num_layers = len(kernel_sizes)
		self.conv_layers = nn.ModuleList()
		# self.weights = nn.ModuleList()
		# self.biases = nn.ModuleList()
		
		for i, filter_size in enumerate(kernel_sizes):
			self.conv_layers.append(cnn_layer(in_channels=emb_size, out_channels=num_filter, 
											  kernel_size=kernel_sizes[i], stride=1, 
											  padding=0, bias=True))
	
	def forward(self, input):
		outputs = []
		# pdb.set_trace()
		for i in range(self.num_layers):
			output = self.conv_layers[i](input)	# (n_words, n_chars-filter_size+1, n_filters)
			pooled = torch.max(output, dim=2)[0]	# channel is dim1.
			outputs.append(pooled)
		return torch.cat(outputs, 1)
	

class cnn_layer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
		super(cnn_layer, self).__init__()
		self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
									kernel_size=kernel_size, stride=stride, 
									padding=padding, bias=bias)
		self.relu = torch.nn.ReLU()
	def forward(self, input):
		return self.relu(self.conv(input))