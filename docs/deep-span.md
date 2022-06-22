# Deep Span Representations for Named Entity Recognition

This page describes the materials and code for "Deep Span Representations for Named Entity Recognition". 

This software is for REVIEW USE ONLY. Please DO NOT DISTRIBUTE. 


## Setup
### Install dependencies
```bash
$ conda install numpy=1.18.5 pandas=1.0.5 xlrd=1.2.0 matplotlib=3.2.2 
$ conda install pytorch=1.7.1 torchvision=0.8.2 torchtext=0.8.1 {cpuonly|cudatoolkit=10.2} -c pytorch 
$ pip install -r requirements.txt 
```

### Install `eznlp`
* From source (suggested)
```bash
$ python setup.py sdist
$ pip install dist/eznlp-<version>.tar.gz --no-deps
```


### Download and process datasets
* English datasets
    * ACE 2004: Download from https://catalog.ldc.upenn.edu/LDC2005T09; Process following Lu and Roth (2015).
    * ACE 2005: Download from https://catalog.ldc.upenn.edu/LDC2006T06; Process following Lu and Roth (2015).
    * GENIA: Download from http://www.geniaproject.org/genia-corpus; Process following Lu and Roth (2015).
    * KBP 2017: Download from https://catalog.ldc.upenn.edu/LDC2017D55; Process following Lin et al. (2019) and Shen et al. (2022).
    * CoNLL 2003.
    * OntoNotes 5: Download from https://catalog.ldc.upenn.edu/LDC2013T19; Process following Pradhan et al. (2013).

* Chinese datasets
    * Weibo NER: Download from https://github.com/hltcoe/golden-horse.
    * Resume NER: Download from https://github.com/jiesutd/LatticeLSTM.


### Download pretrained language models
Download the pretrained language models by `transformers` and save to `assets/transformers`.

```python
import transformers

for model_name in ["roberta-base", "hfl/chinese-macbert-base"]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    model = transformers.AutoModelForPreTraining.from_pretrained(model_name)
    model.save_pretrained(f"assets/transformers/{model_name}")
```


## Running the Code
For English datasets:

```bash
$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --dataset {ace2004 | ace2005 | genia | kbp2017 | conll2003 | conll2012} \
    --doc_level \
    --pre_subtokenize \
    --num_epochs 50 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --batch_size 48 \
    --ck_decoder specific_span \
    --affine_dim 300 \
    --sb_epsilon 0.1 \
    --sse_no_share_weights_ext \
    --sse_no_share_interm2 \
    --sse_max_span_size {10 | 15 | 20 | 25} \
    --bert_arch RoBERTa_base \
    --use_interm2 \
    --hid_dim 400 \
    [options]
```

For Chinese datasets:

```bash

$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --dataset {WeiboNER | ResumeNER} \
    --pre_merge_enchars \
    --num_epochs 50 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --batch_size 48 \
    --ck_decoder specific_span \
    --affine_dim 300 \
    --sb_epsilon 0.1 \
    --sse_no_share_weights_ext \
    --sse_no_share_interm2 \
    --sse_max_span_size {10 | 15 | 20 | 25} \
    --bert_arch MacBERT_base \
    --use_interm2 \
    --hid_dim 400 \
    [options]
```


See more details for options: 

```bash
$ python scripts/entity_recognition.py --help
```
