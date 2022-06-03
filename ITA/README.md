# ITA: Image-Text Alignments for Multi-Modal Named Entity Recognition

The code is for our NAACL 2022 paper: [ITA: Image-Text Alignments for Multi-Modal Named Entity Recognition](https://arxiv.org/pdf/2112.06482.pdf).

**I**mage-**T**ext **A**lignments (ITA) aligns image features into the textual space so that the textual transformer-based embeddings can be fully utilized to model the interactions between images and texts.

## Guide

- [Datasets](#datasets)
- [Training](#training)
- [Citing Us](#Citing-Us)

## Datasets

**Our preprocessed datasets**: [[OneDrive]](https://1drv.ms/u/s!Am53YNAPSsodhPEnSbEMTnsmdsiceg?e=f55rbp)

### Extract Texts on Your Own Datasets

- **VinVL Features and Object Labels**: please follow the guide in [scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark#vinvl-feature-extraction).

- **Image Captions**: given the VinVL features, follow the guide in [VinVL](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md) to extract the image captions. 

- **OCR Texts**: please follow the guide in [Tesseract](https://github.com/tesseract-ocr/tesseract).

## Training

Following the guide in [KB-NER](https://github.com/Alibaba-NLP/KB-NER#testing) the `data_folder` should be modified for each config files.

To train our model with **cross-view alignment**, run:
```bash
# Twitter 2015
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_twitter15_doc_joint_multiview_posterior_4temperature_captionobj_classattr_vinvl_ocr_ner24.yaml
```
```bash
# Twitter 2017
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_twitter17_doc_joint_multiview_posterior_2temperature_captionobj_classattr_vinvl_ocr_ner23.yaml
```
```bash
# snap
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_snap_doc_joint_multiview_posterior_4temperature_captionobj_classattr_vinvl_ocr_ner24.yaml
```



To train our model without **cross-view alignment**, run:
```bash
# Twitter 2015
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_twitter15_doc_captionobj_classattr_vinvl_ocr_ner23.yaml
```
```bash
# Twitter 2017
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_twitter17_doc_captionobj_classattr_vinvl_ocr_ner25.yaml
```
```bash
# snap
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_snap_doc_captionobj_classattr_vinvl_ocr_ner25.yaml
```


## Citing Us
If you feel the code helpful, please cite:
```
@inproceedings{wang2022ita,
    title = "{{ITA: Image-Text Alignments for Multi-Modal Named Entity Recognition}}",
    author = "Wang, Xinyu and Gui, min and Jiang, Yong and Jia, Zixia and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei",
    booktitle = "{2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics}",
    address = "Online",
    month = jul,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
