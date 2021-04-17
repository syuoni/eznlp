# Easy Natural Language Processing

This repository consists of:
* `eznlp.text_classification`
* `eznlp.sequence_tagging`
* `eznlp.language_modeling`


## Experiment Results

### Text Classification 
| Dataset      | Language | Best Acc. | Model Specification |
|:------------:|:--------:|:---------:|:-------------------:|
| IMDb         | English  | 95.78     | RoBERTa-base + Attention |
| Yelp Full    | English  | 71.55     | RoBERTa-base + Attention |
| Yelp 2013    | English  | 70.80     | RoBERTa-base + Attention |
| ChnSentiCorp | Chinese  | 95.83     | BERT-base + Attention    |
| THUCNews-10  | Chinese  | 98.98     | RoBERTa-base + Attention |

See [Text Classification](docs/text_classification.md) for more details. 


### Named Entity Recognition
| Dataset      | Language | Best F1 | Model Specification |
|:------------:|:--------:|:-------:|:-------------------:|
| CoNLL 2003   | English  | 93.26   | RoBERTa-large + LSTM + CRF |
| OntoNotes v5 | English  | 91.05   | RoBERTa-base + LSTM + CRF  |
| MSRA         | Chinese  | 93.13*  | BERT + LSTM + CRF          |
| WeiboNER v2  | Chinese  | 70.48   | BERT + LSTM + CRF          |
| ResumeNER    | Chinese  | 95.97   | BERT + LSTM + CRF          |

See [Named Entity Recognition](docs/entity_recognition.md) for more details. 


## Future Plans
- [x] SoftLexicon
- [ ] Radical-Level Features
- [x] Experiments on Chinese NER datasets
- [x] Experiments on text classification datasets
- [ ] Focal loss (and combined to CRF?)
- [ ] Relation Extraction
- [ ] Span-based models (e.g., SpERT)
- [ ] Data Augmentation
