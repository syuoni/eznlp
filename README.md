# Easy Natural Language Processing

`eznlp` is a `PyTorch`-based package for neural natural language processing, currently supporting:
* Text classification
* Named Entity Recognition
    * Sequence tagging
    * Span classification
    * Boundary Selection
* Relation extraction
* Attribute extraction


## Experiment Results
### Text Classification 
| Dataset      | Language | Our Best Imp. Acc. | Model Specification |
|:------------:|:--------:|:------------------:|:-------------------:|
| IMDb         | English  | 95.78         | RoBERTa-base + Attention |
| Yelp Full    | English  | 71.55         | RoBERTa-base + Attention |
| Yelp 2013    | English  | 70.80         | RoBERTa-base + Attention |
| ChnSentiCorp | Chinese  | 95.83         | BERT-base + Attention    |
| THUCNews-10  | Chinese  | 98.98         | RoBERTa-base + Attention |

See [Text Classification](docs/text_classification.md) for more details. 


### Named Entity Recognition
| Dataset      | Language | Our Best Imp. F1 | Model Specification |
|:------------:|:--------:|:----------------:|:-------------------:|
| CoNLL 2003   | English  | 93.26     | RoBERTa-large + LSTM + CRF |
| OntoNotes v5 | English  | 91.05     | RoBERTa-base + LSTM + CRF  |
| MSRA         | Chinese  | 96.18     | BERT + LSTM + CRF          |
| WeiboNER v2  | Chinese  | 70.48     | BERT + LSTM + CRF          |
| ResumeNER    | Chinese  | 95.97     | BERT + LSTM + CRF          |
| OntoNotes v4 | Chinese  | 82.29     | BERT + LSTM + CRF          |
| OntoNotes v5 | Chinese  | 80.31     | BERT + LSTM + CRF          |

See [Named Entity Recognition](docs/entity_recognition.md) for more details. 


### Relation Extraction
| Dataset      | Language | Our Best Imp. F1 | Model Specification |
|:------------:|:--------:|:----------------:|:-------------------:|
| CoNLL 2004   | English  | 89.17 / 75.03    | SpERT (with RoBERTa-base + LSTM) |
| SciERC       | English  | 69.29 / 36.65    | SpERT (with RoBERTa-base)        |

See [Relation Extraction](docs/relation_extraction.md) for more details. 


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


## Citation
If you find our code useful, please cite the following paper: 

```
@article{zhu2021framework,
  title={A Unified Framework of Medical Information Annotation and Extraction for {Chinese} Clinical Text},
  author={Zhu, Enwei and Sheng, Qilin and Yang, Huanwan and Li, Jinpeng},
  journal={Working Paper},
  year={2021}
}
```


## Future Plans
- [x] SoftLexicon
- [ ] Radical-Level Features
- [x] Experiments on Chinese NER datasets
- [x] Experiments on text classification datasets
- [x] Focal loss (and combined to CRF?)
- [ ] Dice loss
- [x] Relation Extraction
- [x] Span-based models (e.g., SpERT)
- [ ] NER / RE as MRC
- [ ] Pair selection (multi-head selection; RE for flat entities)
- [ ] Data Augmentation
- [ ] LR finder for MSRA