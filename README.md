# KB-NER: a Knowledge-based System for Multilingual Complex Named Entity Recognition

The code is for the winner system (DAMO-NLP) of SemEval 2022 MultiCoNER shared task over 10 out of 13 tracks. [[Rankings]](https://multiconer.github.io/results), [[Paper]](https://arxiv.org/pdf/2203.00545.pdf).

KB-NER is a knowledge-based system, where we build a multilingual knowledge base based on Wikipedia to provide related context information to the NER model. 

<!-- ![image](https://user-images.githubusercontent.com/17926734/157019580-ec411abe-92ff-4144-a591-78b4b1d2f26c.png) ![image](https://user-images.githubusercontent.com/17926734/157019621-b3ff9734-f669-4c9b-80eb-8a70c75fd6fc.png | width=0.5\textwidth) -->

<!-- <img src="https://user-images.githubusercontent.com/17926734/157019580-ec411abe-92ff-4144-a591-78b4b1d2f26c.png" width=500> <img src="https://user-images.githubusercontent.com/17926734/157019621-b3ff9734-f669-4c9b-80eb-8a70c75fd6fc.png" width=500> -->

![1646656832(1)](https://user-images.githubusercontent.com/17926734/157036466-289323ff-c57e-45d0-b960-50e12dea78e9.jpg)


## Guide

- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Datasets](#datasets)
  - [Trained Models](#Trained-Models)
- [Building Knowledge-based System](#Building-Knowledge-based-System)
	- [Knowledge Base Building](#Knowledge-Base-Building)
	  - [Index Building](#Index-Building)
	  - [Retrieval-based Data Augmentation](#Retrieval-based-Data-Augmentation)
	  - [Context Processing](#Context-Processing)
	- [Multi-stage Fine-tuning](#Multi-stage-Fine-tuning)
	- [Majority Voting Ensemble](#Majority-Voting-Ensemble)
    - [(Optional) CE and ACE Models](#Optional-CE-and-ACE-Models)
- [Config File](#Config-File)
- [Citing Us](#Citing-Us)
- [Acknowledgement](#Acknowledgement)
- [Contact](#contact)

## Requirements
To run our code, install:

```
pip install -r requirements.txt
```

## Quick Start

### Datasets

To ease the code

Training and development data with retrieved knowledge: [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodhO59bmCs05MulenL3Q?e=itf8yQ)

Test data with retrieved knowledge: [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodhO5-DrWR8jbDhVoZSQ?e=Xe7uwL)

Our model predictions submitted at the test phase: [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodhO58074pzX2JMMvfrQ?e=ytwYsB). We believe the predictions can be used for distilling knowledge from our system.


#### Recommended Training and Testing Data for Each Language

| Language/Track | Training | Testing|
| -------------------------------  | ---  | ----------- | 
| EN-English | EN-English_conll_rank_eos_doc_full_wiki_v3 | EN-English_conll_rank_eos_doc_full_wiki_v3_test |
| ES-Spanish | ES-Spanish_conll_rank_eos_doc_full_wiki_v3 | ES-Spanish_conll_rank_eos_doc_full_wiki_v3_test |
| NL-Dutch | NL-Dutch_conll_rank_eos_doc_full_wiki_v3 | NL-Dutch_conll_rank_eos_doc_full_wiki_v3_test |
| RU-Russian | RU-Russian_conll_rank_eos_doc_full_wiki_v3 | RU-Russian_conll_rank_eos_doc_full_wiki_v3_test |
| TR-Turkish | TR-Turkish_conll_rank_eos_doc_full_wiki_v3 | TR-Turkish_conll_rank_eos_doc_full_wiki_v3_test |
| KO-Korean | KO-Korean_conll_rank_eos_doc_full_wiki_v3 | KO-Korean_conll_rank_eos_doc_full_wiki_v3_test |
| FA-Farsi | FA-Farsi_conll_rank_eos_doc_full_wiki_v3 | FA-Farsi_conll_rank_eos_doc_full_wiki_v3_test |
| DE-German | DE-German_conll_rank_eos_doc_full_wiki_v3_sentence_withent | DE-German_conll_rank_eos_doc_full_wiki_v3_test_sentence_withent |
| ZH-Chinese | ZH-Chinese_conll_rank_eos_doc_full_wiki_v3_sentence | ZH-Chinese_conll_rank_eos_doc_full_wiki_v3_test_sentence |
| HI-Hindi | HI-Hindi_conll_rank_eos_doc_full_wiki_v3_sentence | HI-Hindi_conll_rank_eos_doc_full_wiki_v3_test_sentence |
| BN-Bangla | BN-Bangla_conll_rank_eos_doc_full_wiki_v3_sentence | BN-Bangla_conll_rank_eos_doc_full_wiki_v3_test_sentence |
| MULTI-Multilingual | All monolingual datasets `*_conll_rank_eos_doc_full_wiki_v3` | MULTI_Multilingual_conll_rank_eos_doc_full_wiki_v3_test_langwiki |
| MIX-Code_mixed | MIX-Code_mixed_conll_rank_eos_doc_full_wiki_v3_sentence | MIX-Code_mixed_conll_rank_eos_doc_full_wiki_v3_test_sentence |
| MIX-Code_mixed (Iterative) | MIX-Code_mixed_conll_rank_eos_doc_full_wiki_v4_sentence_withent | MIX-Code_mixed_conll_rank_eos_doc_full_wiki_v4_test_sentence_withent |

The meanning of the suffixes in the folder names are listed as follows:

| Suffix | Meaning |
| -------------------------------  | ---  | 
| `test` | Our test data with retrieved contexts from knowledge base|
| `v3` | Contexts in the data are from sentence retrieval|
| `v4` | Contexts in the data are from iterative entity retrieval| 
| `sentence` | Using matched sentences as the contexts (`Wiki-Sent<sub>-link</sub>` in the paper)|
| `sentence_withent` | Using matched sentences with wiki anchors as the contexts (`Wiki-Sent` in the paper)|
| w/o `sentence` and `sentence_withent` | Using matched paragraphs with wiki anchors as the contexts (`Wiki-Para` in the paper)|

Note that in iterative entity retrieval datasets, the training data are using gold entities to retrieve knowledge while the test data (with 'test' in folder name) are using predicted entities (by our ensembled models based on sentence retrieval) for retrieval.

### Trained Models

Since there 130+ trained models for our submission in the test phase, we only release our trained models for English (monolingual), Multilingual and Code-mixed. 

Download link: [Uploading ...](), you may follow [Training and Testing on MultiCoNER Datasets](#Training-and-Testing-on-MultiCoNER-Datasets) to select the required trained models for downloading.

- Put the downloaded models into `resources/taggers`.

---

### Training and Testing on MultiCoNER Datasets

This section is a guide for running our code with our trained models and processed datasts downloaded above. If you want to train and make predictions on your own datasets, please refer to [Building Knowledge-based System](#Building-Knowledge-based-System) for the guide about how to build the knowledge retrieval system and training your own knowledge-based models from scratch.

#### Testing

We provide four trained models for the three tracks, which are the candidates of the ensemble models of our submission.
To make prediction with our trained models, run:

**For English**, `xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner20` is required. Change the `data_folder` in the yaml file to the path to your downloaded datasets for all the configs. For example:
```yaml
ColumnCorpus-EN-EnglishDOC:
    column_format:
      0: text
      1: pos
      2: upos
      3: ner
    comment_symbol: '# id'
    data_folder: EN-English_conll_rank_eos_doc_full_wiki_v3 # change the data_folder at here
    tag_to_bioes: ner
```

Run:
```bash
# English
CUDA_VISIBLE_DEVICES=0 python -u train_with_teacher.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner20.yaml --parse --keep_order --target_dir EN-English_conll_rank_eos_doc_full_wiki_v3_test --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 
```

**For Multilingual**, `xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner24` is required.

Run:
```bash
# Multilingual
CUDA_VISIBLE_DEVICES=0 python -u train_with_teacher.py --config config/xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner24.yaml --parse --keep_order --target_dir MULTI_Multilingual_conll_rank_eos_doc_full_wiki_v3_test_langwiki --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 
```

**For Code-mixed**, 

`xlmr-large-pretuned-tuned-wiki-full-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_sentence_ner40` is the model based on sentence retrieval and

`xlmr-large-pretuned-tuned-wiki-full-v4-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v4_sentence_withent_ner30` is the model based on iterative entity retrieval.

The sentence-retrieval-based models are used for predict entity mentions for the iterative entity retrieval. The iterative-entity-retrieval-based models are expected to be stronger than the sentence-retrieval-based models in code-mixed.

Run:
```bash
# Code-mixed + Sentence Retrieval
CUDA_VISIBLE_DEVICES=0 python -u train_with_teacher.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_sentence_ner40.yaml --parse --keep_order --target_dir MIX_Code_mixed_conll_rank_eos_doc_full_wiki_v3_test_sentence_withent --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 

# Code-mixed + Iterative Entity Retrieval
CUDA_VISIBLE_DEVICES=0 python -u train_with_teacher.py --config config/xlmr-large-pretuned-tuned-wiki-full-v4-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v4_sentence_withent_ner30.yaml --parse --keep_order --target_dir MIX_Code_mixed_conll_rank_eos_doc_full_wiki_v4_test_sentence_withent --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 
```

---

#### Training the monolingual models

To train the monolingual model based on fine-tuned multilingual embeddings, the trained model `xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_10upsample_addmix_ner23` is required. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner20.yaml
```

If you want to train the other languages, change configurations in the config. For example:
```yaml
model_name: xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_es_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner20 # change the model_name to let the model be saved at a new folder, here we change _en_ to _es_ to represent Spanish.
...
  ColumnCorpus-ES-SpanishDOC: # Training dataset settings
    column_format:
      0: text
      1: pos
      2: upos
      3: ner
    comment_symbol: '# id'
    data_folder: ES-Spanish_conll_rank_eos_doc_full_wiki_v3 # change the dataset folder
    tag_to_bioes: ner
  Corpus: ColumnCorpus-ES-SpanishDOC # It must be the same as the corpus name above
```
Please refer to [this table](#Recommended-Training-and-Testing-Data-for-Each-Language) to decide the training dataset for each language.

**Note:** this model and the following two model are trained on both the training and development sets. As a result, the development F1 score should be about 100 during training.

#### Training the multilingual models

Training multilingual models need to download `xlm-roberta-large-ft10w`, which is a continue pretrained model over the shared task data. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner24.yaml
```

#### Training the code-mixed models

For our code-mixed models with sentence retrieval, download `xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_10upsample_addmix_ner23` and Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_sentence_ner40.yaml
```

For our code-mixed models with iterative entity retrieval, download `xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v4_10upsample_addmix_ner23` and Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-pretuned-tuned-wiki-full-v4-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v4_sentence_withent_ner30.yaml
```

---

## Building Knowledge-based System

### Knowledge Base Building

#### Index Building

Our wiki-based retrieval system is built on [ElasticSearch](https://www.elastic.co/), and you need to install ElasticSearch properly before you can build a knowledge base. For a tutorial on installation, please refer to this [link](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/targz.html).

First you need to download the latest version of wiki dumps from [Wikimedia](https://www.wikimedia.org/) and store them in the lmdb database. You can run the following commands:

```bash
cd kb/dumps
./download.sh
./convert_db.sh
```

Then convert the files from xml format to plain text. Please run the command:

```bash
cd ..
./parse_text.sh
```

Finally you can build the knowledge base through ElasticSearch, i.e. create indexes for 11 languages ( Note that you need to make sure that ElasticSearch is running and listening to the default port 9200 ):

```bash
./bulid_kb.sh
```

#### Retrieval-based Data Augmentation

We provide two types of retrieval-based augmentations, one at the sentence level and one at the entity level. First you need to place the datasets in conll format under `kb/datasets`. Then run the following code as needed.

+ Sentence-level retrieval augmentation:

  ```bash
  python generate_data.py --lan en
  ```

+ Entity-level retrieval augmentation:

  ```bash
  python generate_data.py --lan en --with_entity
  ```

Note that `--lan` specifies the language and `--with_entity` indicates whether to retrieve at entity level (default is false). 

The retrieval results are presented in the following format. The first line is the original sentence and the entities in the sentence. Next are the 10 (default) most relevant retrieval results, one per row.

```
original sentence \t entity #1 | entity #2 ···
retrieved sentence #1 \t associated paragraph #1 \t associated title #1 \t score #1 \t wiki url #1 \t hits on the sentence #1 ---#--- hits on the title #1
retrieved sentence #2 \t associated paragraph #2 \t associated title #2 \t score #2 \t wiki url #2 \t hits on the sentence #2 ---#--- hits on the title #2
···
retrieved sentence #10 \t associated paragraph #10 \t associated title #10 \t score #10 \t wiki url #10 \t hits on the sentence #10 ---#---  hits on the title #10
```

Let's show an example:

```
anthology is a compilation album by new zealand singer songwriter and multi instrumentalist bic runga . compilation album | new zealand | bic runga 
Anthology is a compilation album by New Zealand singer-songwriter and multi-instrumentalist Bic Runga.	Anthology is a <e:Compilation album>compilation album</e> by <e:New Zealand>New Zealand</e> singer-songwriter and multi-instrumentalist <e:Bic Runga>Bic Runga</e>. The album was initially set to be released on 23 November 2012, but ultimately released on 1 December 2012 in New Zealand. The album cover was revealed on 29 October 2012.	Anthology (Bic Runga album)	145.28241	https://en.wikipedia.org/wiki/Anthology (Bic Runga album)	<hit>Anthology</hit> <hit>is</hit> <hit>a</hit> <hit>compilation</hit> <hit>album</hit> <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> <hit>singer</hit>-<hit>songwriter</hit> <hit>and</hit> <hit>multi</hit>-<hit>instrumentalist</hit> <hit>Bic</hit> <hit>Runga</hit> ---#--- Anthology (<hit>Bic</hit> <hit>Runga</hit> <hit>album</hit>)
···
```



If you want to do iterative retrieval at entity level, please convert the model prediction to conll format and then perform entity level retrieval.



#### Context Processing

We consider the retrieved paragraphs as context and concatenate them after the original sentence.



---

### Multi-stage Fine-tuning

Taking the transferring from multilingual models to monolingual models as an examle, firstly we train a multilingual model (which is the same model as [Training the multilingual models](#Training-the-multilingual-models) but is trained only on the training data):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner24.yaml
```
In the config file, you can find:
```yaml
...
train:
  ...
  save_finetuned_embedding: true
  ...
...
```
The code will save the fine-tuned embeddings at the end of each epoch when `save_finetuned_embedding` is set to `true`. In this case, you can find the saved embeddings at `resources/taggers/xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner24/xlm-roberta-large-ft10w`.

Then, you can use the fine-tuned `xlm-roberta-large-ft10w` embeddings as the initialization of XLM-R embeddings, taking fine-tuning the English monolingual model as an example, set the embeddings in `config/xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner20.yaml` as:
```yaml
embeddings:
  TransformerWordEmbeddings-0:
    fine_tune: true
    layers: '-1'
    model: resources/taggers/xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner24/xlm-roberta-large-ft10w
    pooling_operation: first
```
Run the model training:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner20.yaml
```

---

### Majority Voting Ensemble

We provide an example of majority voting ensemble. Download the all English predictions at [here](https://1drv.ms/u/s!Am53YNAPSsodhO5_6HzQRzANirhltg?e=V6X6nw) and run:
```bash
python ensemble_prediction.py 
```

---

### (Optional) CE and ACE Models

We also provide code for running CE and ACE (in Section 5.6 of the paper) for monolingual models, you can check the guide at [here](https://github.com/Alibaba-NLP/ACE#Instructions-for-Reproducing-Results). To select embedding candidates, besides the embeddings listed in [ACE embedding list](https://github.com/Alibaba-NLP/ACE#download-embeddings), we recommend to use the fine-tuned XLM-R embeddings checkpoints for the multilingual models and monolingual models. `config/xlmr-task-wiki-extdoc_en-xlmr-task-wiki-extdoc_multi-xlmr-pretuned-wiki-tuned_word_flair_mflair_55b-elmo_150epoch_32batch_0.1lr_1000hidden_en_crf_reinforce_doc_freeze_norelearn_nodev_amz_wiki_v3_ner6.yaml` is an example for ACE configuration we tried during system building:

```yaml
embeddings:
  ELMoEmbeddings-0: #ELMo
    options_file: elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
    weight_file: elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
  FastWordEmbeddings-0:
    embeddings: en
    freeze: true
  FlairEmbeddings-0: #Flair
    model: en-forward
  FlairEmbeddings-1:
    model: en-backward
  FlairEmbeddings-2:
    model: multi-forward
  FlairEmbeddings-3:
    model: multi-backward
  TransformerWordEmbeddings-0: #Fine-tuned XLM-R trained on English dataset
    layers: '-1'
    model: resources/taggers/xlmr-large-pretuned-tuned-wiki-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner24/xlm-roberta-large
    pooling_operation: first
    use_internal_doc: true
  TransformerWordEmbeddings-1: #Fine-tuned RoBERTa trained on English dataset
    layers: '-1'
    model: resources/taggers/en-xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_5000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_saving_finetune_amz_doc_wiki_v3_ner25/roberta-large
    pooling_operation: first
    use_internal_doc: true
  TransformerWordEmbeddings-2: #Fine-tuned XLMR trained on Multilingual dataset
    layers: '-1'
    model: resources/taggers/xlmr-large-pretuned-tuned-new-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_saving_amz_doc_wiki_v3_ner24/xlm-roberta-large
    pooling_operation: first

```

To run CE, you can change config like this: 
```yaml
train:
  ...
  max_episodes: 1
  max_epochs: 300
  ...
```

---
## Config File

You can find the description of each part in the config file at [here](https://github.com/Alibaba-NLP/CLNER#config-file).

## Citing Us
If you feel the code helpful, please cite:
```
@article{wang2022damonlp,
      title={{DAMO-NLP at SemEval-2022 Task 11: A Knowledge-based System for Multilingual Named Entity Recognition}}, 
      author={Xinyu Wang and Yongliang Shen and Jiong Cai and Tao Wang and Xiaobin Wang and Pengjun Xie and Fei Huang and Weiming Lu and Yueting Zhuang and Kewei Tu and Wei Lu and Yong Jiang},
      year={2022},
      eprint={2203.00545},
      url= {https://arxiv.org/abs/2112.06482},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{wang2021improving,
    title = "{{Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning}}",
    author = "Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei",
    booktitle = "{the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (\textbf{ACL-IJCNLP 2021})}",
    address = "Online",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

If you feel the CE and ACE models helpful:
```
@inproceedings{wang2020automated,
    title = "{{Automated Concatenation of Embeddings for Structured Prediction}}",
    author = "Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei",
    booktitle = "{the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (\textbf{ACL-IJCNLP 2021})}",
    month = aug,
    address = "Online",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}

@inproceedings{wang-etal-2020-embeddings,
    title = "{More Embeddings, Better Sequence Labelers?}",
    author = "Wang, Xinyu  and
      Jiang, Yong  and
      Bach, Nguyen  and
      Wang, Tao  and
      Huang, Zhongqiang  and
      Huang, Fei  and
      Tu, Kewei",
    booktitle = "{{\bf EMNLP-Findings 2020}}",
    month = nov,
    year = "2020",
    address = "Online",
    % publisher = "Association for Computational Linguistics",
    % url = "https://www.aclweb.org/anthology/2020.findings-emnlp.356",
    % doi = "10.18653/v1/2020.findings-emnlp.356",
    pages = "3992--4006",
}
```

## Acknowledgement

Start from the great repo [flair version 0.4.3](https://github.com/flairNLP/flair), the code has been modified a lot. This code also supports our previous work such as multilingual knowledge distillation ([MultilangStructureKD](https://github.com/Alibaba-NLP/MultilangStructureKD)), automated concatenation of embeddings ([ACE](https://github.com/Alibaba-NLP/ACE)) and utilizing external contexts ([CLNER](https://github.com/Alibaba-NLP/CLNER)). You can also try these approaches in this repo.

## Contact 

Feel free to email any questions or comments to issues or to [Xinyu Wang](http://wangxinyu0922.github.io/).

For questions about the knowledge retrieval module, you can also ask [Yongliang Shen](mailto:syl@zju.edu.cn) and [Jiong Cai](mailto:caijiong@shanghaitech.edu.cn)
