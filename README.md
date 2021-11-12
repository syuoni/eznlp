# Easy Natural Language Processing

`eznlp` is a `PyTorch`-based package for neural natural language processing, currently supporting:
* Text Classification
* Named Entity Recognition
    * Sequence Tagging
    * Span Classification
    * Boundary Selection
* Relation Extraction
* Attribute Extraction
* Machine Translation
* Image Captioning


## Experimental Results
### Text Classification 
| Dataset      | Language | Our Best Acc. | Model |
|:------------:|:--------:|:------------------:|:-------------------:|
| IMDb         | English  | 95.78         | RoBERTa-base + Attention |
| Yelp Full    | English  | 71.55         | RoBERTa-base + Attention |
| Yelp 2013    | English  | 70.80         | RoBERTa-base + Attention |
| ChnSentiCorp | Chinese  | 95.83         | BERT-base + Attention    |
| THUCNews-10  | Chinese  | 98.98         | RoBERTa-base + Attention |

See [Text Classification](docs/text-classification.pdf) for more details. 


### Named Entity Recognition
| Dataset      | Language | Our Best F1 | Model |
|:------------:|:--------:|:----------------:|:-------------------:|
| CoNLL 2003   | English  | 93.26     | RoBERTa-large + LSTM + CRF |
| OntoNotes 5  | English  | 91.05     | RoBERTa-base + LSTM + CRF  |
| MSRA         | Chinese  | 96.18     | BERT + LSTM + CRF          |
| WeiboNER v2  | Chinese  | 70.48     | BERT + LSTM + CRF          |
| ResumeNER    | Chinese  | 95.97     | BERT + LSTM + CRF          |
| OntoNotes 4  | Chinese  | 82.29     | BERT + LSTM + CRF          |
| OntoNotes 5  | Chinese  | 80.31     | BERT + LSTM + CRF          |

See [Named Entity Recognition](docs/entity-recognition.pdf) for more details. 


### Relation Extraction
| Dataset      | Language | Our Best F1 <br>(Ent / Rel / Rel+) | Model |
|:------------:|:--------:|:----------------:|:-------------------:|
| CoNLL 2004   | English  | 89.17 / -     / 75.03 | SpERT (w/ RoBERTa-base + LSTM) |
| SciERC       | English  | 69.29 / 48.93 / 36.65 | SpERT (w/ RoBERTa-base)        |

See [Relation Extraction](docs/relation-extraction.pdf) for more details. 


## Installation
### With `pip`
```bash
$ pip install eznlp
```

### From source
```bash
$ python setup.py sdist
$ pip install dist/eznlp-<version>.tar.gz
```


## Running the Code
### Text classification
```bash
$ python scripts/text_classification.py --dataset <dataset> [options]
```

### Entity recognition
```bash
$ python scripts/entity_recognition.py --dataset <dataset> [options]
```

### Relation extraction
```bash
$ python scripts/relation_extraction.py --dataset <dataset> [options]
```

### Attribute extraction
```bash
$ python scripts/attribute_extraction.py --dataset <dataset> [options]
```
