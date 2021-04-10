# Easy Natural Language Processing

This repository consists of:
* `eznlp.text_classification`
* `eznlp.sequence_tagging`
* `eznlp.language_modeling`

## Experiments
### Text Classification (English)
Configurations of our implementation:
* Word embeddings are initialized with GloVe
* `From-scratch` models
    * Optimizer: Adadelta (lr=0.5)
    * Batch size: 64
    * Number of epochs: 50
* `Fine-tuining` models
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=1e-5)
    * Batch size: 32
    * Number of epochs: 10
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * BERT/RoBERTa models are loaded with dropout rate of 0.2

#### IMDb
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | -                    | -     | 91.58 | num_layers=1 |
| LSTM + Attention         | McCann et al. (2017) | 91.1  | 92.09 | num_layers=1 |
| BERT-base + Attention    | Sun et al. (2019)    | 94.60 | 
| RoBERTa-base + Attention | -                    | -     | 

#### Yelp Full
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | Zhang et al. (2015)  | 58.17 | 65.97 | num_layers=2 |
| LSTM + Attention         | -                    | -     | 68.61 | num_layers=2 |
| BERT-base + Attention    | Sun et al. (2019)    | 69.94 | 
| RoBERTa-base + Attention | -                    | -     | 

#### Yelp 2013 (with User and Product IDs)
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | Chen et al. (2016)   | 62.7  | 64.96 | num_layers=2 |
| LSTM + Attention         | Chen et al. (2016)   | 63.1  | 64.84 | num_layers=2 |
| BERT-base + Attention    | -                    | -     | 68.76 |
| RoBERTa-base + Attention | -                    | -     | 70.80 |


### Named Entity Recognition (English)
Configurations of our implementation:
* Tagging scheme: BIOES
* Word embeddings are initialized with GloVe
* `From-scratch` models
    * Optimizer: SGD (lr=0.1)
    * Batch size: 32
    * Number of epochs: 100
* `Fine-tuining` models
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=1e-5)
    * Batch size: 48
    * Number of epochs: 50
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * BERT/RoBERTa models are loaded with dropout rate of 0.2
    * BERT-uncased models inputs are converted to "truecase"
    
♦ use both training and development splits for training. 

#### CoNLL 2003 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| CharLSTM + LSTM + CRF       | Lample et al. (2016)  | 90.94         | 91.28 | num_layers=1 |
| CharCNN + LSTM + CRF        | Ma and Hovy (2016)    | 91.21         | 90.70 | num_layers=1 |
| ELMo + Char + LSTM + CRF    | Peters et al. (2018)  | 92.22 (0.10)  | 92.60 | num_layers=1 |
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)   | 93.09♦ (0.12) | 92.60 | num_layers=1 |
| ELMo + Flair + Char + LSTM + CRF | -                | -             | 92.67 | num_layers=1 |
| BERT-base + Softmax         | Devlin et al. (2018)  | 92.4          | 92.02 | 
| BERT-base + CRF             | -                     | -             | 92.38 | 
| BERT-base + LSTM + CRF      | -                     | -             | 92.40 | 
| BERT-large + Softmax        | Devlin et al. (2018)  | 92.8          | 92.34 | 
| BERT-large + CRF            | -                     | -             | 92.64 | 
| BERT-large + LSTM + CRF     | -                     | -             | 92.80 | 
| RoBERTa-base + Softmax      | Liu et al. (2019)     | -             | 92.39 | 
| RoBERTa-base + CRF          | -                     | -             | 92.59 | 
| RoBERTa-base + LSTM + CRF   | -                     | -             | 92.71 | 
| RoBERTa-large + Softmax     | Liu et al. (2019)     | -             | 92.81 | 
| RoBERTa-large + CRF         | -                     | -             | 93.20 | 
| RoBERTa-large + LSTM + CRF  | -                     | -             | 93.26 | 

#### OntoNotes v5 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| CharLSTM + LSTM + CRF       | Lample et al. (2016)    | -            | 87.68 | num_layers=2 |
| CharCNN + LSTM + CRF        | Chiu and Nichols (2016) | 86.17 (0.22) | 87.43 | num_layers=2 |
| ELMo + Char + LSTM + CRF    | Peters et al. (2018)    | -            | 89.71 | num_layers=2 |
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)     | 89.3♦        | 89.02 | num_layers=2 |
| ELMo + Flair + Char + LSTM + CRF | -                  | -            | 89.55 | num_layers=2 |
| BERT-base + Softmax         | Devlin et al. (2018)    | -            | 89.35 | 
| BERT-base + CRF             | -                       | -            | 90.14 | 
| BERT-base + LSTM + CRF      | -                       | -            | 89.89 | 
| RoBERTa-base + Softmax      | Liu et al. (2019)       | -            | 90.22 | 
| RoBERTa-base + CRF          | -                       | -            | 90.83 | 
| RoBERTa-base + LSTM + CRF   | -                       | -            | 91.05 | 

### Named Entity Recognition (Chinese)
Configurations of our implementation:
* Character-based
* Tagging scheme: BIOES
* `From-scratch` models
    * Optimizer: AdamW (lr=1e-3)
    * Batch size: 32
    * Number of epochs: 100
* `Fine-tuining` models
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=1e-5)
    * Batch size: 48
    * Number of epochs: 50
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * BERT/RoBERTa models are loaded with dropout rate of 0.2

#### SIGHAN 2006 / MSRA 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 88.81 | 89.49 | num_layers=2 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 91.87 | 92.02 | num_layers=2 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 93.18 |
| FLAT + CRF                  | Li et al. (2019)      | 94.35 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 93.66 | 93.64 | num_layers=2; Adamax (lr=1e-3) |
| BERT + CRF                  | Ma et al. (2020)      | 93.76 | 93.16 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 94.83 | 93.13 |
| FLAT + BERT + CRF           | Li et al. (2019)      | 96.09 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 95.42 |

#### WeiboNER v2 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 52.77 | 50.19 | num_layers=2 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 56.75 | 57.18 | num_layers=2 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 58.79 |
| FLAT + CRF                  | Li et al. (2019)      | 63.42 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 61.42 | 61.17 | num_layers=2; Adamax (lr=5e-3) |
| BERT + CRF                  | Ma et al. (2020)      | 63.80 | 68.79 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 67.33 | 70.48 |
| FLAT + BERT + CRF           | Li et al. (2019)      | 68.55 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 70.50 |

#### ResumeNER 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 93.48 | 94.93 | num_layers=2 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 94.41 | 94.51 | num_layers=2 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 94.46 |
| FLAT + CRF                  | Li et al. (2019)      | 94.93 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 95.53 | 95.48 | num_layers=2; Adamax (lr=2e-3) | 
| BERT + CRF                  | Ma et al. (2020)      | 95.68 | 95.68 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 95.51 | 95.97 |
| FLAT + BERT + CRF           | Li et al. (2019)      | 95.86 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 96.11 |

### Relation Extraction


## Future Plans
- [x] SoftLexicon
- [ ] Radical-Level Features
- [x] Experiments on Chinese NER datasets
- [x] Experiments on text classification datasets
- [ ] Focal loss (and combined to CRF?)
- [ ] Relation Extraction
- [ ] Span-based models (e.g., SpERT)
- [ ] Data Augmentation


## References
* Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level convolutional networks for text classification. *NIPS 2015*.
* Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., and Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT 2016*.
* Ma, X., and Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. *ACL 2016*.
* Chiu, J. P., and Nichols, E. (2016). Named Entity Recognition with Bidirectional LSTM-CNNs. *TACL*, 4:357-370.
* Chen, H., Sun, M., Tu, C., Lin, Y., and Liu, Z. (2016). Neural sentiment classification with user and product attention. *EMNLP 2016*.
* McCann, B., Bradbury, J., Xiong, C., and Socher, R. (2017). Learned in translation: Contextualized word vectors. *NIPS 2017*.
* Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. (2018). Deep Contextualized Word Representations. *NAACL-HLT 2018*.
* Akbik, A., Blythe, D., and Vollgraf, R. (2018). Contextual String Embeddings for Sequence Labeling. *COLING 2018*.
* Akbik, A., Bergmann, T., and Vollgraf, R. (2019). Pooled Contextualized Embeddings for Named Entity Recognition. *NAACL-HLT 2019*.
* Zhang, Y., and Yang, J. (2018). Chinese NER Using Lattice LSTM. *ACL 2018*.
* Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. 
* Sun, C., Qiu, X., Xu, Y., and Huang, X. (2019). How to fine-tune BERT for text classification? *CCL 2019*.
* Li, X., Yan, H., Qiu, X., and Huang, X. J. (2020). FLAT: Chinese NER using flat-lattice transformer. *ACL 2020*.
* Ma, R., Peng, M., Zhang, Q., Wei, Z., and Huang, X. J. (2020). Simplify the Usage of Lexicon in Chinese NER. *ACL 2020*.
