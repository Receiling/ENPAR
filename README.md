# PSPE
Source code of EACL2021 paper ["ENPAR: Enhancing Entity and Entity Pair Representations for Joint Entity Relation Extraction."](https://www.aclweb.org/anthology/2021.eacl-main.251.pdf).

## Requirements
* `python`: 3.7.6
* `pytorch`: 1.4.0
* `spacy`: 2.0.12
* `transformers`: 2.8.0
* `configargparse`: 1.1
* `bidict`: 0.18.0
* `fire`: 0.2.1

## Pre-training
Before pre-training, please prepare a pre-training corpus (e.g. Wikipedia), the format of the pre-training corpus must be the same as the file [`data/pretrain/wikipedia_raw.txt`](https://github.com/Receiling/ENPAR/blob/master/data/pretrain/wikipedia_raw.txt), and replace the file `data/pretrain/wikipedia_raw.txt`.

Then run the script ['data_preprocess.sh'](https://github.com/Receiling/ENPAR/blob/master/data_preprocess.sh) to preprocess the pre-training corpus. We provide an example of the final processed dataset in the folder `data/pretrain/`.

Pre-training:
```bash
$ python entity_relation_extractor_pretrain.py \
                            --config_file pretrain.yml \
                            --device 0 \
                            --fine_tune
```

## Fine-tuning
Before fine-tuning, please download the pre-trained model [`ENPAR`](https://pan.baidu.com/s/1ice6IkZFBQhSl8LMhf-Chg)(password: 2imb), and place the pre-trained model in the folder main folder. And make sure that the format of the dataset must be the same as [`data/demo/train.json`](https://github.com/Receiling/ENPAR/blob/master/data/demo/train.json).
```bash 
python entity_relation_extractor.py \
                        --config_file finetune.yml \
                        --device 0 \
                        --fine_tune
```

## Cite
If you find our code is useful, please cite:
```
@inproceedings{wang-etal-2021-enpar,
    title = "{ENPAR}:Enhancing Entity and Entity Pair Representations for Joint Entity Relation Extraction",
    author = "Wang, Yijun  and
      Sun, Changzhi  and
      Wu, Yuanbin  and
      Zhou, Hao  and
      Li, Lei  and
      Yan, Junchi",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.251",
    pages = "2877--2887",
    abstract = "Current state-of-the-art systems for joint entity relation extraction (Luan et al., 2019; Wad-den et al., 2019) usually adopt the multi-task learning framework. However, annotations for these additional tasks such as coreference resolution and event extraction are always equally hard (or even harder) to obtain. In this work, we propose a pre-training method ENPAR to improve the joint extraction performance. ENPAR requires only the additional entity annotations that are much easier to collect. Unlike most existing works that only consider incorporating entity information into the sentence encoder, we further utilize the entity pair information. Specifically, we devise four novel objectives,i.e., masked entity typing, masked entity prediction, adversarial context discrimination, and permutation prediction, to pre-train an entity encoder and an entity pair encoder. Comprehensive experiments show that the proposed pre-training method achieves significant improvement over BERT on ACE05, SciERC, and NYT, and outperforms current state-of-the-art on ACE05.",
}
```


