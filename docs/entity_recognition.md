## Named Entity Recognition (English)
Configurations of our implementation:
* Tagging scheme: BIOES
* Word embeddings are initialized with GloVe
* `From-scratch` models of `sequence tagging`
    * Optimizer: SGD (lr=0.1)
    * Batch size: 32
    * Number of epochs: 100
* `From-scratch` models of `span classification`
    * Optimizer: Adadelta (lr=1.0)
    * Batch size: 64
    * Number of epochs: 100
* `Fine-tuining` models
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=1e-5)
    * Batch size: 48
    * Number of epochs: 50
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * BERT/RoBERTa models are loaded with dropout rate of 0.2
    * BERT-uncased models inputs are converted to "truecase"
    
♦ use both training and development splits for training ([Biaffine](https://github.com/juntaoy/biaffine-ner/issues/16#issuecomment-716492521)).  

### CoNLL 2003 
| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
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

| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| SpERT (with CharLSTM + LSTM)| -                     | -             | 91.22 | num_layers=2 |
| SpERT (with BERT-base)    | Eberts and Ulges (2019) | -             | 91.97 | 
| SpERT (with BERT-base + LSTM) | -                   | -             | 92.62 | 
| SpERT (with RoBERTa-base)        | -                | -             | 92.36 | 
| SpERT (with RoBERTa-base + LSTM) | -                | -             | 92.50 | 

| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| Biaffine (with CharLSTM + LSTM)     | -                | -          | 91.05 | num_layers=2 |
| Biaffine (with BERT-base)           | -                | -          | 92.47 | 
| Biaffine (with BERT-base + LSTM)    | -                | -          | 92.74 | 
| Biaffine (with BERT-large)          | Yu et al. (2020) | 93.5♦      | 92.67 |
| Biaffine (with RoBERTa-base)        | -                | -          | 92.56 | 
| Biaffine (with RoBERTa-base + LSTM) | -                | -          | 92.77 | 
| Biaffine (with RoBERTa-large)       | -                | -          | 93.26 |

### OntoNotes v5 
| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| CharLSTM + LSTM + CRF       | Lample et al. (2016)    | -            | 87.68 | num_layers=2 |
| CharCNN + LSTM + CRF        | Chiu and Nichols (2016) | 86.17 (0.22) | 87.43 | num_layers=2 |
| ELMo + Char + LSTM + CRF    | Peters et al. (2018)    | -            | 89.71 | num_layers=2 |
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)     | 89.3♦        | 89.02 | num_layers=2 |
| ELMo + Flair + Char + LSTM + CRF | -                  | -            | 89.55 | num_layers=2 |
| BERT-base + Softmax         | Devlin et al. (2018)    | -            | 89.35 | 
| BERT-base + CRF             | -                       | -            | 90.14 | 
| BERT-base + LSTM + CRF      | -                       | -            | 89.89 | 
| Biaffine (with BERT-large)  | Yu et al. (2020)        | 91.3♦        | -     |
| RoBERTa-base + Softmax      | Liu et al. (2019)       | -            | 90.22 | 
| RoBERTa-base + CRF          | -                       | -            | 90.83 | 
| RoBERTa-base + LSTM + CRF   | -                       | -            | 91.05 | 



## Named Entity Recognition (Chinese)
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
    * BERT models are loaded with dropout rate of 0.2
    * BERT refers to BERT-wwm (Cui et al., 2019)

### MSRA (SIGHAN 2006) 
* All experiments use testing split as development split ([SoftLexicon](https://github.com/v-mipeng/LexiconAugmentedNER/issues/3#issuecomment-634563407)). 

| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 88.81 | 89.49 | num_layers=2 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 91.87 | 92.02 | num_layers=2 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 93.18 |
| FLAT + CRF                  | Li et al. (2019)      | 94.35 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 93.66 | 93.64 | num_layers=2; Adamax (lr=1e-3) |
| BERT + CRF                  | Ma et al. (2020)      | 93.76 | 95.92 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 94.83 | 96.18 |
| FLAT + BERT + CRF           | Li et al. (2019)      | 96.09 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 95.42 |

### WeiboNER v2 
| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 52.77 | 50.19 | num_layers=2 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 56.75 | 57.18 | num_layers=2 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 58.79 |
| FLAT + CRF                  | Li et al. (2019)      | 63.42 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 61.42 | 61.17 | num_layers=2; Adamax (lr=5e-3) |
| BERT + CRF                  | Ma et al. (2020)      | 63.80 | 68.79 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 67.33 | 70.48 |
| FLAT + BERT + CRF           | Li et al. (2019)      | 68.55 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 70.50 |

### ResumeNER 
| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 93.48 | 94.93 | num_layers=2 |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 94.41 | 94.51 | num_layers=2 |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 94.46 |
| FLAT + CRF                  | Li et al. (2019)      | 94.93 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 95.53 | 95.48 | num_layers=2; Adamax (lr=2e-3) | 
| BERT + CRF                  | Ma et al. (2020)      | 95.68 | 95.68 |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 95.51 | 95.97 |
| FLAT + BERT + CRF           | Li et al. (2019)      | 95.86 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 96.11 |

### OntoNotes v4 
| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| LSTM + CRF                  | Zhang and Yang (2018) | 64.30 |  |  |
| Bichar + LSTM + CRF         | Zhang and Yang (2018) | 71.89 |  |  |
| Lattice-LSTM + CRF          | Zhang and Yang (2018) | 73.88 |
| FLAT + CRF                  | Li et al. (2019)      | 76.45 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)      | 75.64 |  |  | 
| BERT + CRF                  | Ma et al. (2020)      | 77.93 |  |
| BERT + LSTM + CRF           | Ma et al. (2020)      | 81.82 |  |
| FLAT + BERT + CRF           | Li et al. (2019)      | 81.82 |
| SoftLexicon + BERT + CRF    | Ma et al. (2020)      | 82.81 |

### OntoNotes v5 
| Model | Paper | Reported F1 | Our Imp. F1 | Notes |
|:-----:|:-----:|:-----------:|:-----------:|:-----:|
| LSTM + CRF                  |                   | -     | 73.30 | num_layers=2 |
| Bichar + LSTM + CRF         |                   | -     | 75.36 | num_layers=2 |
| Lattice-LSTM + CRF          | Jie and Lu (2019) | 76.67 |
| SoftLexicon + LSTM + CRF    | Ma et al. (2020)  | -     | 76.13 | num_layers=2; Adamax (lr=2e-3) | 
| BERT + CRF                  |                   | -     | 80.34 |
| BERT + LSTM + CRF           |                   | -     | 80.31 |


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
* Jie, Z., and Lu, W. (2019). Dependency-Guided LSTM-CRF for Named Entity Recognition. *EMNLP 2019*.
* Eberts, M., and Ulges, A. (2019). Span-based joint entity and relation extraction with Transformer pre-training. *ECAI 2020*.
* Cui, Y., Che, W., Liu, T., Qin, B., Yang, Z., Wang, S., and Hu, G. (2019). Pre-training with whole word masking for Chinese BERT. *EMNLP 2020*. 
* Yu, J., Bohnet, B., and Poesio, M. (2020). Named entity recognition as dependency parsing. *ACL 2020*. 
* Li, X., Yan, H., Qiu, X., and Huang, X. J. (2020). FLAT: Chinese NER using flat-lattice transformer. *ACL 2020*.
* Ma, R., Peng, M., Zhang, Q., Wei, Z., and Huang, X. J. (2020). Simplify the Usage of Lexicon in Chinese NER. *ACL 2020*.
