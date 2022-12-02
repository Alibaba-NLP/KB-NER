# KB-NER: a Knowledge-based System for Multilingual Complex Named Entity Recognition

The code is for the winner system (DAMO-NLP) of SemEval 2022 MultiCoNER shared task over 10 out of 13 tracks. [[Rankings]](https://multiconer.github.io/multiconer_1/results), [[Paper]](https://arxiv.org/pdf/2203.00545.pdf).

KB-NER is a knowledge-based system, where we build a multilingual knowledge base based on Wikipedia to provide related context information to the NER model. 

![1646656832(1)](https://user-images.githubusercontent.com/17926734/157036466-289323ff-c57e-45d0-b960-50e12dea78e9.jpg)

## News
 - **2022-11**: [AdaSeq](https://github.com/modelscope/AdaSeq): An all-in-one and easy-to-use library for developing sequence understanding models is released. 
- **2022-07** Our [paper](https://arxiv.org/pdf/2203.00545.pdf) wins the **Best System Paper Award** (top 0.45%=1/221) at SemEval 2022!
- **2022-05** Check our [ITA](https://github.com/Alibaba-NLP/KB-NER/tree/main/ITA) for multimodal named entity recognition!

## Guide

- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Datasets](#datasets)
  - [Trained Models](#Trained-Models)
  - [Training and Testing on MultiCoNER Datasets](#training-and-testing-on-multiconer-datasets) 
- [Building Knowledge-based System](#Building-Knowledge-based-System)
	- [Knowledge Base Building](#Knowledge-Base-Building)
	  - [Index Building](#Index-Building)
	  - [Retrieval-based Data Augmentation](#Retrieval-based-Data-Augmentation)
	  - [Context Processing](#Context-Processing)
	- [Multi-stage Fine-tuning](#Multi-stage-Fine-tuning)
	- [Majority Voting Ensemble](#Majority-Voting-Ensemble)
    - [(Optional) CE and ACE Models](#Optional-CE-and-ACE-Models)
- [Predict files](#Predict-files)
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

To ease the code running, please download our pre-processed datasets.

- **Training and development data with retrieved knowledge**: [[OneDrive]](https://1drv.ms/u/s!Am53YNAPSsodhO59bmCs05MulenL3Q?e=itf8yQ)

- **Test data with retrieved knowledge**: [[OneDrive]](https://1drv.ms/u/s!Am53YNAPSsodhO5-DrWR8jbDhVoZSQ?e=Xe7uwL)

- **Our model predictions submitted at the test phase**: [[OneDrive]](https://1drv.ms/u/s!Am53YNAPSsodhO58074pzX2JMMvfrQ?e=ytwYsB). We believe the predictions can be used for distilling knowledge from our system.


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

The meaning of the suffixes in the folder names are listed as follows:

| Suffix | Description |
| -------------------------------  | ---  | 
| `test` | Our test data with retrieved contexts from knowledge base|
| `v3` | Contexts in the data are from sentence retrieval|
| `v4` | Contexts in the data are from iterative entity retrieval| 
| `sentence` | Using matched sentences as the contexts (**Wiki-Sent<sub>-link</sub>** in the paper)|
| `sentence_withent` | Using matched sentences with wiki anchors as the contexts (**Wiki-Sent** in the paper)|
| w/o `sentence` and `sentence_withent` | Using matched paragraphs with wiki anchors as the contexts (**Wiki-Para** in the paper)|

Note that in iterative entity retrieval datasets, the training data are using gold entities to retrieve knowledge while the test data (with 'test' in the folder name) are using predicted entities (by our ensembled models based on sentence retrieval) for retrieval.

### Trained Models

Since there are 130+ trained models for our submission in the test phase, we only release our trained models for English (monolingual), Multilingual and Code-mixed. 

- **Download link**: [[OneDrive]](https://1drv.ms/u/s!Am53YNAPSsodhO8VUU-Bd4NE5Wb9SQ?e=w84xWo), you may follow [Training and Testing on MultiCoNER Datasets](#training-and-testing-on-multiconer-datasets) to select the required trained models for downloading.
  + Put the downloaded models into `resources/taggers`.

---

### Training and Testing on MultiCoNER Datasets

This section is a guide for running our code with our trained models and processed datasets downloaded above. If you want to train and make predictions on your own datasets, please refer to [Building Knowledge-based System](#Building-Knowledge-based-System) for the guide about how to build the knowledge retrieval system and train your own knowledge-based models from scratch.

#### Testing

We provide four trained models for the three tracks, which are the candidates of the ensemble models of our submission.
To make predictions with our trained models, run:

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
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner20.yaml --parse --keep_order --target_dir EN-English_conll_rank_eos_doc_full_wiki_v3_test --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 
```

**For Multilingual**, `xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner24` is required.

Run:
```bash
# Multilingual
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/xlmr-large-pretuned-tuned-wiki-first_3epoch_1batch_4accumulate_0.000005lr_10000lrrate_multi_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_ner24.yaml --parse --keep_order --target_dir MULTI_Multilingual_conll_rank_eos_doc_full_wiki_v3_test_langwiki --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 
```

**For Code-mixed**, 

`xlmr-large-pretuned-tuned-wiki-full-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_sentence_ner40` is the model based on sentence retrieval and

`xlmr-large-pretuned-tuned-wiki-full-v4-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v4_sentence_withent_ner30` is the model based on iterative entity retrieval.

The sentence-retrieval-based models are used to predict entity mentions for the iterative entity retrieval. The iterative-entity-retrieval-based models are expected to be stronger than the sentence-retrieval-based models in code-mixed.

Run:
```bash
# Code-mixed + Sentence Retrieval
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/xlmr-large-pretuned-tuned-wiki-full-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v3_sentence_ner40.yaml --parse --keep_order --target_dir MIX_Code_mixed_conll_rank_eos_doc_full_wiki_v3_test_sentence_withent --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 

# Code-mixed + Iterative Entity Retrieval
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/xlmr-large-pretuned-tuned-wiki-full-v4-first_100epoch_1batch_4accumulate_0.000005lr_10000lrrate_mix_monolingual_crf_fast_norelearn_sentbatch_sentloss_withdev_finetune_saving_amz_doc_wiki_v4_sentence_withent_ner30.yaml --parse --keep_order --target_dir MIX_Code_mixed_conll_rank_eos_doc_full_wiki_v4_test_sentence_withent --num_columns 4 --batch_size 32 --output_dir semeval2022_predictions 
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

**Note:** this model and the following two models are trained on both the training and development sets. As a result, the development F1 score should be about 100 during training.

#### Training the multilingual models

`xlm-roberta-large-ft10w` is required to train the multilingual models, which is a continue pretrained model over the shared task data. Run:
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

Our wiki-based retrieval system is built on [ElasticSearch](https://www.elastic.co/), and you need to install ElasticSearch properly before building knowledge bases. For a tutorial on installation, please refer to this [link](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/targz.html). In addition, in order to make ElasticSearch support Chinese word segmentation, we recommend you to install [`ik-analyzer`](https://github.com/medcl/elasticsearch-analysis-ik).

After installing ElasticSearch, you can build your local multilingual wiki knowledge bases. First you need to download the latest version of wiki dumps from [Wikimedia](https://www.wikimedia.org/) and store them in the lmdb database. You can run the following commands:

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

We provide two types of retrieval-based augmentations, one at the sentence level and one at the entity level. First you need to place the datasets in CoNLL format under `kb/datasets`. Then run the following code as needed.

+ Sentence-level retrieval augmentation:

  ```bash
  python generate_data.py --lan en
  ```

+ Entity-level retrieval augmentation:

  ```bash
  python generate_data.py --lan en --with_entity
  ```

Note that `--lan` specifies the language and `--with_entity` indicates whether to retrieve at the entity level (default is false). 

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
anthology is a compilation album by new zealand singer songwriter and multi instrumentalist bic runga .	compilation album | new zealand | bic runga 
Anthology is a compilation album by New Zealand singer-songwriter and multi-instrumentalist Bic Runga.	Anthology is a <e:Compilation album>compilation album</e> by <e:New Zealand>New Zealand</e> singer-songwriter and multi-instrumentalist <e:Bic Runga>Bic Runga</e>. The album was initially set to be released on 23 November 2012, but ultimately released on 1 December 2012 in New Zealand. The album cover was revealed on 29 October 2012.	Anthology (Bic Runga album)	145.28241	https://en.wikipedia.org/wiki/Anthology (Bic Runga album)	<hit>Anthology</hit> <hit>is</hit> <hit>a</hit> <hit>compilation</hit> <hit>album</hit> <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> <hit>singer</hit>-<hit>songwriter</hit> <hit>and</hit> <hit>multi</hit>-<hit>instrumentalist</hit> <hit>Bic</hit> <hit>Runga</hit> ---#--- Anthology (<hit>Bic</hit> <hit>Runga</hit> <hit>album</hit>)
Briolette Kah Bic Runga  (born 13 January 1976), recording as Bic Runga, is a New Zealand singer-songwriter and multi-instrumentalist pop artist.	Briolette Kah Bic Runga  (born 13 January 1976), recording as Bic Runga, is a New Zealand singer-songwriter and multi-instrumentalist pop artist. Her first three <e:Album>studio albums</e> debuted at number one on the <e:Recorded Music NZ>New Zealand Top 40 Album charts</e>. Runga has also found success internationally in Australia, Ireland and the United Kingdom with her song "<e:Sway (Bic Runga song)>Sway</e>".	Bic Runga	125.18798	https://en.wikipedia.org/wiki/Bic Runga	Briolette Kah <hit>Bic</hit> <hit>Runga</hit>  (born 13 January 1976), recording as <hit>Bic</hit> <hit>Runga</hit>, <hit>is</hit> <hit>a</hit> <hit>New</hit> <hit>Zealand</hit> <hit>singer</hit>-<hit>songwriter</hit> ---#--- <hit>Bic</hit> <hit>Runga</hit>
Birds is the third studio album by New Zealand artist Bic Runga.	Birds is the third <e:Album>studio album</e> by <e:New Zealand>New Zealand</e> artist <e:Bic Runga>Bic Runga</e>. The album was released in New Zealand on 28 November 2005. The album was Bic's third no.1 album garnering platinum status in its first week. The album was certified 3x platinum. The album won the <e:Aotearoa Music Award for Album of the Year>New Zealand Music Award for Album of the Year</e> in 2006, her second award for Best Album, after her debut release <e:Drive (Bic Runga album)>Drive</e>.	Birds (Bic Runga album)	100.14264	https://en.wikipedia.org/wiki/Birds (Bic Runga album)	Birds <hit>is</hit> the third studio <hit>album</hit> <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> artist <hit>Bic</hit> <hit>Runga</hit>. ---#--- Birds (<hit>Bic</hit> <hit>Runga</hit> <hit>album</hit>)
"Sway" is a song by New Zealand singer Bic Runga.	"Sway" is a song by New Zealand singer <e:Bic Runga>Bic Runga</e>. It was released as the second single from her debut studio album, <e:Drive (Bic Runga album)>Drive</e> (1997), in 1997. The song peaked at  7 in New Zealand and No. 10 in Australia, earning gold <e:Music recording certification>certifications</e> in both countries. At the <e:Aotearoa Music Awards>32nd New Zealand Music Awards</e>, the song won three awards: Single of the Year, Best Songwriter, and Best Engineer (Simon Sheridan). In 2001, it was voted the <e:APRA Top 100 New Zealand Songs of All Time#6>6th best New Zealand song of all time</e> by members of <e:APRA AMCOS>APRA</e>. A music video directed by John Taft was made for the song.	Sway (Bic Runga song)	97.816284	https://en.wikipedia.org/wiki/Sway (Bic Runga song)	"Sway" <hit>is</hit> <hit>a</hit> song <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> <hit>singer</hit> <hit>Bic</hit> <hit>Runga</hit>. ---#--- Sway (<hit>Bic</hit> <hit>Runga</hit> song)
Drive is the debut solo album by New Zealand artist Bic Runga, released on 14 July 1997.	Drive is the debut solo album by New Zealand artist <e:Bic Runga>Bic Runga</e>, released on 14 July 1997. The album went seven times <e:Music recording certification>platinum</e> in New Zealand, and won the <e:Aotearoa Music Award for Album of the Year>New Zealand Music Award for Album of the Year</e> at the <e:Aotearoa Music Awards>32nd New Zealand Music Awards</e>.	Drive (Bic Runga album)	94.014656	https://en.wikipedia.org/wiki/Drive (Bic Runga album)	Drive <hit>is</hit> the debut solo <hit>album</hit> <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> artist <hit>Bic</hit> <hit>Runga</hit>, released on 14 July 1997. ---#--- Drive (<hit>Bic</hit> <hit>Runga</hit> <hit>album</hit>)
Bic Runga at Discogs	Bic Runga at <e:Discogs>Discogs</e>	Bic Runga	93.661865	https://en.wikipedia.org/wiki/Bic Runga	<hit>Bic</hit> <hit>Runga</hit> at Discogs ---#--- <hit>Bic</hit> <hit>Runga</hit>
Close Your Eyes is the fifth studio album by New Zealand singer-song writer Bic Runga.	Close Your Eyes is the fifth studio album by <e:New Zealand>New Zealand</e> singer-song writer <e:Bic Runga>Bic Runga</e>. The album is made up of ten covers and two original tracks. Upon announcement of the album in October, Runga said: "There are so many songs I've always wanted to cover. I wanted to see if I could not just be a singer-songwriter, but someone who could also interpret songs. In the process, I found there are so many reasons why a cover version wouldn't work, perhaps because the lyrics were not something I could relate to first hand, because technically I wasn't ready or because the original was too iconic. But the songs that all made it on the record specifically say something about where I'm at in my life, better than if I'd written it myself. It was a challenging process, I'm really proud of the singing and the production and the statement".	Close Your Eyes (Bic Runga album)	90.77379	https://en.wikipedia.org/wiki/Close Your Eyes (Bic Runga album)	Close Your Eyes <hit>is</hit> the fifth studio <hit>album</hit> <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> <hit>singer</hit>-song writer <hit>Bic</hit> <hit>Runga</hit>. ---#--- Close Your Eyes (<hit>Bic</hit> <hit>Runga</hit> <hit>album</hit>)
All tracks by Bic Runga.	All tracks by <e:Bic Runga>Bic Runga</e>.	Drive (Bic Runga album)	89.630394	https://en.wikipedia.org/wiki/Drive (Bic Runga album)	All tracks <hit>by</hit> <hit>Bic</hit> <hit>Runga</hit>. ---#--- Drive (<hit>Bic</hit> <hit>Runga</hit> <hit>album</hit>)
"Sorry" is a song by New Zealand recording artist, Bic Runga.	"Sorry" is a song by New Zealand recording artist, <e:Bic Runga>Bic Runga</e>. The single was released in <e:Australia>Australia</e> and <e:Germany>Germany</e> only as the final single from her debut studio album, <e:Drive (Bic Runga album)>Drive</e> (1997).	Sorry (Bic Runga song)	89.33654	https://en.wikipedia.org/wiki/Sorry (Bic Runga song)	"Sorry" <hit>is</hit> <hit>a</hit> song <hit>by</hit> <hit>New</hit> <hit>Zealand</hit> recording artist, <hit>Bic</hit> <hit>Runga</hit>. ---#--- Sorry (<hit>Bic</hit> <hit>Runga</hit> song)
In November 2008, Runga released Try to Remember Everything which is a collection of unreleased, new and rare Bic Runga recordings from 1996 to 2008.	In November 2008, Runga released <e:Try to Remember Everything>Try to Remember Everything</e> which is a collection of unreleased, new and rare Bic Runga recordings from 1996 to 2008. The album was certified Gold in New Zealand on 14 December 2008, selling over 7,500 copies.	Bic Runga	89.24142	https://en.wikipedia.org/wiki/Bic Runga	In November 2008, <hit>Runga</hit> released Try to Remember Everything which <hit>is</hit> <hit>a</hit> collection of unreleased, <hit>new</hit> ---#--- <hit>Bic</hit> <hit>Runga</hit>
```



If you want to do iterative retrieval at entity level, please convert the model predictions to conll format and then perform entity level retrieval.



#### Context Processing

Here we take `mix` as an example to generate contexts for the datasets.

Usage:
```
$ python kb/context_process.py -h
usage: context_process.py [-h] [--retrieval_file RETRIEVAL_FILE]
                [--conll_folder CONLL_FOLDER] [--lang LANG] [--use_sentence]
                [--use_paragraph_entity]

optional arguments:
  -h, --help            show this help message and exit
  --retrieval_file RETRIEVAL_FILE
                        The retrieved contexts from the knowledge base.
  --conll_folder CONLL_FOLDER
                        The data folder you want to generate contexts, the
                        code will read train, dev, test data in the folder in
                        conll formatting.
  --lang LANG           The language code of the data, for example "en". We
                        have specical processing for Chinese ("zh") and Code-
                        mixed ("mix").
  --use_sentence        use matched sentence in the retrieval results as the
                        contexts
  --use_paragraph_entity
                        use matched sentence and the wiki anchor in the
                        retrieval results as the contexts
```


Given the retrieved contexts and the conll data folder, run (generate contexts for **Wiki-Para**):
```bash
python kb/context_process.py --retrieval_file semeval_retrieve_res/mix.conll --conll_folder semeval_test/MIX_Code_mixed --lang mix
```


To generate contexts for **Wiki-Sent<sub>-link</sub>**, run:
```bash
python kb/context_process.py --retrieval_file semeval_retrieve_res/mix.conll --conll_folder semeval_test/MIX_Code_mixed --lang mix --use_sentence
```

To generate contexts for **Wiki-Sent**, run:
```bash
python kb/context_process.py --retrieval_file semeval_retrieve_res/mix.conll --conll_folder semeval_test/MIX_Code_mixed --lang mix --use_sentence --use_paragraph_entity
```

+ **Note:** the file `semeval_retrieve_res/mix.conll` is the retrieval results for all the sets in `MIX_Code_mixed`. You may need to modify the code to satisfy your own requirements. For more details, you may read [line 972-1002](https://github.com/Alibaba-NLP/KB-NER/blob/63c1351c6cb4e7ce0041c2396d84e866f7d0daa3/kb/context_process.py#L972-L1002) and [line 1006-1029](https://github.com/Alibaba-NLP/KB-NER/blob/63c1351c6cb4e7ce0041c2396d84e866f7d0daa3/kb/context_process.py#L1006-L1029).

---

### Multi-stage Fine-tuning

Taking the transferring from multilingual models to monolingual models as an example, firstly we train a multilingual model (which is the same model as [Training the multilingual models](#Training-the-multilingual-models) but is trained only on the training data):
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

We provide an example of majority voting ensembling. Download the all English predictions at [here](https://1drv.ms/u/s!Am53YNAPSsodhO5_6HzQRzANirhltg?e=V6X6nw) and run:
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

To run CE, you can change the config like this: 
```yaml
train:
  ...
  max_episodes: 1
  max_epochs: 300
  ...
```
---

## Predict files

If you want to predict a certain file, add `train` in the file name and put the file in a certain `$dir` (for example, `parse_file_dir/train.your_file_name`). Run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config $config_file --parse --target_dir $dir --keep_order --batch_size $batch_size --output_dir $output_dir
```

If `$output_dir` is not specified the output file will be `outputs/`
<!-- The format of the file should be `column_format={0: 'text', 1:'ner'}` for sequence labeling or you can modifiy line 337 in `train.py`. The parsed results will be in `outputs/`. -->
Note that you may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.


---
## Config File

You can find the description of each part in the config file at [here](https://github.com/Alibaba-NLP/CLNER#config-file).

## Citing Us
If you feel the code helpful, please cite:
```
@inproceedings{wang-etal-2022-damo,
    title = "{DAMO}-{NLP} at {S}em{E}val-2022 Task 11: A Knowledge-based System for Multilingual Named Entity Recognition",
    author = "Wang, Xinyu  and
      Shen, Yongliang  and
      Cai, Jiong  and
      Wang, Tao  and
      Wang, Xiaobin  and
      Xie, Pengjun  and
      Huang, Fei  and
      Lu, Weiming  and
      Zhuang, Yueting  and
      Tu, Kewei  and
      Lu, Wei  and
      Jiang, Yong",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.200",
    pages = "1457--1468",
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

Starting from the great repo [flair version 0.4.3](https://github.com/flairNLP/flair), the code has been modified a lot. This code also supports our previous work such as multilingual knowledge distillation ([MultilangStructureKD](https://github.com/Alibaba-NLP/MultilangStructureKD)), automated concatenation of embeddings ([ACE](https://github.com/Alibaba-NLP/ACE)) and utilizing external contexts ([CLNER](https://github.com/Alibaba-NLP/CLNER)). You can also try these approaches in this repo.

## Contact 

Feel free to email any questions or comments to issues or to [Xinyu Wang](http://wangxinyu0922.github.io/).

For questions about the knowledge retrieval module, you can also ask [Yongliang Shen](mailto:syl@zju.edu.cn) and [Jiong Cai](mailto:caijiong@shanghaitech.edu.cn).
