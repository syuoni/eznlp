# Easy Natural Language Processing

This repository consists of:
* `eznlp.text_classification`
* `eznlp.sequence_tagging`
* `eznlp.language_modeling`

## Experiments
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
#### SIGHAN 2006 / MSRA 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 88.81 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 91.87 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 93.18 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 93.66 |
| BERT + CRF                  | Ma et al. (2020)      | 93.76 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 94.83 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 95.42 |

#### WeiboNER v2 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 52.77 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 56.75 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 58.79 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 61.42 |
| BERT + CRF                  | Ma et al. (2020)      | 63.80 | 68.85 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 67.33 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 70.50 |

#### ResumeNER 
| Model | Paper | Reported F1 | Our Implementation F1 | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 93.48 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 94.41 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 94.46 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 95.53 |
| BERT + CRF                  | Ma et al. (2020)      | 95.68 | 95.95 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 95.51 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 96.11 |

#### OntoNotes v5 




## Future Plans
- [ ] SoftLexicon
- [ ] Experiments on Chinese NER datasets
- [ ] Experiments on text classification datasets
- [ ] Focal loss (and combined to CRF?)
- [ ] Relation Extraction
- [ ] Span-based models (e.g., SpERT)
- [ ] Data Augmentation


## References
* Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., and Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT 2016*.
* Ma, X., and Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. *ACL 2016*.
* Chiu, J. P., and Nichols, E. (2016). Named Entity Recognition with Bidirectional LSTM-CNNs. *TACL*, 4:357-370.
* Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. (2018). Deep Contextualized Word Representations. *NAACL-HLT 2018*.
* Akbik, A., Blythe, D., and Vollgraf, R. (2018). Contextual String Embeddings for Sequence Labeling. *COLING 2018*.
* Akbik, A., Bergmann, T., and Vollgraf, R. (2019). Pooled Contextualized Embeddings for Named Entity Recognition. *NAACL-HLT 2019*.
* Zhang, Y., and Yang, J. (2018). Chinese NER Using Lattice LSTM. *ACL 2018*.
* Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. 
* Ma, R., Peng, M., Zhang, Q., Wei, Z., and Huang, X. J. (2020). Simplify the Usage of Lexicon in Chinese NER. *ACL 2020*.
