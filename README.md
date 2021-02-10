# Easy Natural Language Processing

This repository consists of:
* `eznlp.text_classification`
* `eznlp.sequence_tagging`
* `eznlp.language_modeling`


## Named Entiry Recognition
### CoNLL 2003 

| Model    | Paper | Reported F1 | Official Code Re-run | Our Implementation | Notes |
|:--------:|:-----:|:-----------:|:--------------------:|:------------------:|:-----:|
| CharLSTM + LSTM + CRF | Lample et al. (2016) | 90.94 | 90.81 | 91.30 | SGD (lr=0.1) |
| CharCNN + LSTM + CRF  | Ma and Hovy (2016)   | 91.21 | 90.64 | 90.13 |   
| ELMo + CharCNN + LSTM + CRF | Peters et al. (2018) | 92.22 (0.10) | 
| BERT-base + Softmax         | Devlin et al. (2018) | 92.4 |
| BERT-large + Softmax        | Devlin et al. (2018) | 92.8 |
| BERT-base + LSTM + CRF      | 
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)  | 93.09* (0.12) |

\* Trained on both training and development sets. 


