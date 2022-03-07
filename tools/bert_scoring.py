import codecs
import bert_score.score as score
import pdb
import nltk
# f= codecs.open('conll_des_title_rank_full_revised.txt','r', errors='ignore')
# f= codecs.open('wnut16_des_title_rank_full_revised_all.txt','r', errors='ignore')
f= codecs.open('en_retrieval.txt','r', errors='ignore')
# f= codecs.open('test.txt','r', errors='ignore')
# f= codecs.open('tech_des_title_rank_full_revised.txt','r', errors='ignore')

# f= codecs.open('bc5cdr_des_title_rank_full_revised.txt','r', errors='ignore')
# f= codecs.open('test.txt','r', errors='ignore')

def score_edit(cands,refs):
	scores = []
	for idx, cand in enumerate(cands):
		ref = refs[idx]
		score = 1 - nltk.edit_distance(cand,ref)/max([len(ref),len(cand)])
		scores.append(score)
		# pdb.set_trace()
	return [scores]
	pass


google_data=f.readlines()
cands = []
refs = []
wrong_count = 0

for lineid, data in enumerate(google_data):
	res=data.strip().split('\t')
	if len(res)==0:
		continue
	if len(res)==1:
		# pdb.set_trace()
		print(lineid)
		wrong_count+=1
		continue
	words = res[0].split()
	if len(words)>200:
		words = words[:200]
		# chunked_count+=1
		res[0] = ' '.join(words)
	# if len(res)>1:
	# 	pdb.set_trace()
	cands.append(res[0])
	refs.append(res[1])
	# if res[1] not in google_dict:
	# 	google_dict[res[1]]=[]
	# google_dict[res[1]].append(res[0])

print('Processed data')
print('Wrong Sentence Count: ', f'{wrong_count}')
# pdb.set_trace()
google_dict={}

# final_scores = score(cands,refs, model_type = 'xlm-roberta-large',)
# final_scores = score_edit(cands,refs)
final_scores = score(cands,refs, lang = 'en')
F1_scores = final_scores[-1]

# pdb.set_trace()
for idx, cand in enumerate(cands):
	ref = refs[idx]
	F1 = F1_scores[idx]
	if ref not in google_dict:
		google_dict[ref]=[]
	google_dict[ref].append((F1.item(),cand))
	# google_dict[ref].append((F1,cand))
# pdb.set_trace()
# file_writer = open('conll_des_title_bertscore_tfidf_full_revised.txt','w')
file_writer = open('wnut16_des_title_bertscore_full_revised.txt','w')
# file_writer = open('pubmed_des_title_edit_full_revised.txt','w')
# file_writer = open('test_res.txt','w')
# file_writer = open('tech_des_title_bertscore_full_revised.txt','w')
# file_writer = open('ncbi_des_title_bertscore_tfidf_full_revised.txt','w')
# file_writer = open('bc5cdr_des_title_bertscore_full_revised.txt','w')
# file_writer = open('test_idf.txt','w')

for key in google_dict:
	sents = google_dict[key]
	ranked_sents=sorted(sents,reverse=True)
	new_ranked_sents=[]
	for sent in ranked_sents:
		if sent[1] in new_ranked_sents:
			continue
		# elif sent[0]>0.95:
		# 	continue
		else:
			new_ranked_sents.append(sent[1])
	for sent in new_ranked_sents:
		file_writer.write(sent+'\t'+key+'\n')