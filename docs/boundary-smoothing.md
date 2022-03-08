# Boundary Smoothing for Named Entity Recognition

This page describes the materials and code for "Boundary Smoothing for Named Entity Recognition". 


## Setup
### Installation from source
```bash
$ python setup.py sdist
$ pip install dist/eznlp-<version>.tar.gz
```
The dependencies have been specified in `setup.py`. 


### Download and process datasets
* English datasets
    * CoNLL 2003
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

```python
import transformers

for model_name in ["roberta-base", 
                   "roberta-large", 
                   "hfl/chinese-bert-wwm-ext", 
                   "hfl/chinese-macbert-base", 
                   "hfl/chinese-macbert-large"]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    model = transformers.AutoModelForPreTraining.from_pretrained(model_name)
    model.save_pretrained(f"assets/transformers/{model_name}")
```


## Running the Code
For English datasets:

```bash
$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --num_epochs 50 \
    --batch_size 48 \
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
