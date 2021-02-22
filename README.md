# Easy Natural Language Processing

This repository consists of:
* `eznlp.text_classification`
* `eznlp.sequence_tagging`
* `eznlp.language_modeling`


## Named Entiry Recognition
### CoNLL 2003 

| Model    | Paper | Reported F1 | Official Code Re-run | Our Implementation | Notes |
|:--------:|:-----:|:-----------:|:--------------------:|:------------------:|:-----:|
| CharLSTM + LSTM + CRF | Lample et al. (2016) | 90.94 | 90.81 | 91.23 | SGD (lr=0.05); num_layers=1 |
| CharCNN + LSTM + CRF  | Ma and Hovy (2016)   | 91.21 | 90.64 | 90.62 | SGD (lr=0.1); num_layers=1 |
| ELMo + CharCNN + LSTM + CRF | Peters et al. (2018) | 92.22 (0.10)  | - | 92.42 | SGD (lr=0.1) |
| BERT-base + Softmax         | Devlin et al. (2018) | 92.4 | - | 92.12 | AdamW (lr=1e-5); Warmup; Truecase |
| BERT-base + CRF             | -                    | -    | - | 92.00 | AdamW (lr=1e-5); Warmup; Truecase |
| RoBERTa-base + CRF          | -                    | -    | - | 91.08 | AdamW (lr=1e-5); Warmup |
| BERT-base + LSTM + Softmax  | -                    | -    | - | 92.10 | AdamW (lr=2e-5/2e-4); Warmup; Truecase |
| BERT-base + LSTM + CRF      | -                    | -    | - | 92.11 | AdamW (lr=2e-5/2e-4); Warmup; Truecase |
| BERT-large + Softmax        | Devlin et al. (2018) | 92.8 | - |
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)  | 93.09* (0.12) | - | 92.34 | SGD (lr=0.1) |

\* Model trained on both training and development sets. 

