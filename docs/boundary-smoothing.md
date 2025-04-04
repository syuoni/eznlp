# Boundary Smoothing for Named Entity Recognition
This page describes the materials and code for "Boundary Smoothing for Named Entity Recognition". 


## Setup
### Installation
Setup an environment and install the dependencies and `eznlp` according to [README](../README.md).


### Download and process datasets
* English datasets
    * CoNLL 2003.
    * OntoNotes 5: Download from https://catalog.ldc.upenn.edu/LDC2013T19; Process following Pradhan et al. (2013).
    * ACE 2004: Download from https://catalog.ldc.upenn.edu/LDC2005T09; Process following Lu and Roth (2015).
    * ACE 2005: Download from https://catalog.ldc.upenn.edu/LDC2006T06; Process following Lu and Roth (2015).

* Chinese datasets
    * OntoNotes 4: Download from https://catalog.ldc.upenn.edu/LDC2011T03; Process following Che et al. (2013).
    * MSRA: Download from https://github.com/v-mipeng/LexiconAugmentedNER.
    * Weibo NER: Download from https://github.com/hltcoe/golden-horse.
    * Resume NER: Download from https://github.com/jiesutd/LatticeLSTM.


### Download pretrained language models
Download the pretrained language models by `transformers` and save to `assets/transformers`.

```bash
git clone https://huggingface.co/google-bert/bert-base-uncased  assets/transformers/bert-base-uncased
git clone https://huggingface.co/google-bert/bert-base-cased    assets/transformers/bert-base-cased
git clone https://huggingface.co/google-bert/bert-large-uncased assets/transformers/bert-large-uncased
git clone https://huggingface.co/google-bert/bert-large-cased   assets/transformers/bert-large-cased
git clone https://huggingface.co/FacebookAI/roberta-base        assets/transformers/roberta-base
git clone https://huggingface.co/FacebookAI/roberta-large       assets/transformers/roberta-large
git clone https://huggingface.co/hfl/chinese-bert-wwm-ext       assets/transformers/hfl/chinese-bert-wwm-ext
git clone https://huggingface.co/hfl/chinese-macbert-base       assets/transformers/hfl/chinese-macbert-base
git clone https://huggingface.co/hfl/chinese-macbert-large      assets/transformers/hfl/chinese-macbert-large
```


## Running the Code
For English datasets:

```bash
$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --num_epochs 50 \
    --batch_size 48 \
    --num_grad_acc_steps 1 \
    --dataset {conll2003 | conll2012 | ace2004 | ace2005} \
    --ck_decoder boundary_selection \
    --sb_epsilon {0.0 | 0.1 | 0.2 | 0.3} \
    --sb_size {1 | 2} \
    --bert_arch {RoBERTa_base | RoBERTa_large | BERT_base | BERT_large} \
    --use_interm2 \
    [options]
```

For Chinese datasets:

```bash
$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --num_epochs 50 \
    --batch_size 48 \
    --num_grad_acc_steps 1 \
    --dataset {ontonotesv4_zh | SIGHAN2006 | WeiboNER | ResumeNER} \
    --ck_decoder boundary_selection \
    --sb_epsilon {0.0 | 0.1 | 0.2 | 0.3} \
    --sb_size {1 | 2} \
    --bert_arch {BERT_base_wwm | MacBERT_base | MacBERT_large} \
    --use_interm2 \
    [options]
```


See more details for options: 

```bash
$ python scripts/entity_recognition.py --help
```
