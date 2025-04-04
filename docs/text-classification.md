## Text Classification

The experimental results reported here are informal. Please check our published papers for formal results.


### English Datasets

Settings:
* Models w/o PLMs
    * Optimizer: Adadelta (lr=0.5)
    * Batch size: 64
    * Number of epochs: 50
    * Word embeddings are initialized with GloVe
* Models w/ PLMs
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=1e-5)
    * Batch size: 32
    * Number of epochs: 10
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * PLMs are loaded with dropout rate of 0.2

#### IMDb
| Model | Paper | Reported Acc. | Our Imp. Acc. | Notes |
|:-----:|:-----:|:-------------:|:-------------:|:-----:|
| LSTM + MaxPooling        | -                    | -     | 91.58 | num_layers=1 |
| LSTM + Attention         | McCann et al. (2017) | 91.1  | 92.09 | num_layers=1 |
| BERT-base + Attention    | Sun et al. (2019)    | 94.60 | 94.37 |
| RoBERTa-base + Attention | -                    | -     | 95.78 |

#### Yelp Full
| Model | Paper | Reported Acc. | Our Imp. Acc. | Notes |
|:-----:|:-----:|:-------------:|:-------------:|:-----:|
| LSTM + MaxPooling        | Zhang et al. (2015)  | 58.17 | 65.97 | num_layers=2 |
| LSTM + Attention         | -                    | -     | 68.61 | num_layers=2 |
| BERT-base + Attention    | Sun et al. (2019)    | 69.94 | 70.27 |
| RoBERTa-base + Attention | -                    | -     | 71.55 |

#### Yelp 2013 (with User and Product IDs)
| Model | Paper | Reported Acc. | Our Imp. Acc. | Notes |
|:-----:|:-----:|:-------------:|:-------------:|:-----:|
| LSTM + MaxPooling        | Chen et al. (2016)   | 62.7  | 64.96 | num_layers=2 |
| LSTM + Attention         | Chen et al. (2016)   | 63.1  | 64.84 | num_layers=2 |
| BERT-base + Attention    | -                    | -     | 68.76 |
| RoBERTa-base + Attention | -                    | -     | 70.80 |


### Chinese Datasets

Settings:
* Models w/o PLMs
    * Optimizer: Adadelta (lr=1.0)
    * Batch size: 64
    * Number of epochs: 50
    * Word embeddings are initialized by random or with Tencent embeddings (Song et al., 2018)
* Models w/ PLMs
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=2e-5)
    * Batch size: 32
    * Number of epochs: 10
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * PLMs are loaded with dropout rate of 0.2

#### ChnSentiCorp
| Model | Paper | Reported Acc. | Our Imp. Acc. | Notes |
|:-----:|:-----:|:-------------:|:-------------:|:-----:|
| Multi-Channel CNN                      | Liu et al. (2018) | 92.08 |       |
| LSTM + MaxPooling                      | -                 |       | 92.25 | num_layers=2 |
| LSTM + Attention                       | -                 |       | 92.42 | num_layers=2 |
| Tencent Embeddings + LSTM + MaxPooling | -                 |       | 93.50 | num_layers=2 |
| Tencent Embeddings + LSTM + Attention  | -                 |       | 93.08 | num_layers=2 |
| BERT-base + Attention                  | Cui et al. (2019) | 95.3  | 95.83 |
| RoBERTa-base + Attention               | Cui et al. (2019) | 95.8  | 95.08 |

#### THUCNews-10
| Model | Paper | Reported Acc. | Our Imp. Acc. | Notes |
|:-----:|:-----:|:-------------:|:-------------:|:-----:|
| LSTM + MaxPooling                      | -                 |       | 97.66 | num_layers=2 |
| LSTM + Attention                       | -                 |       | 97.24 | num_layers=2 |
| Tencent Embeddings + LSTM + MaxPooling | -                 |       | 98.79 | num_layers=2 |
| Tencent Embeddings + LSTM + Attention  | -                 |       | 98.57 | num_layers=2 |
| BERT-base + Attention                  | Cui et al. (2019) | 97.7  | 98.79 |
| RoBERTa-base + Attention               | Cui et al. (2019) | 97.8  | 98.98 |


### References
* Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *NIPS 2015*.
* Chen, H., Sun, M., Tu, C., Lin, Y., and Liu, Z. (2016). Neural Sentiment Classification with User and Product Attention. *EMNLP 2016*.
* McCann, B., Bradbury, J., Xiong, C., and Socher, R. (2017). Learned in Translation: Contextualized Word Vectors. *NIPS 2017*.
* Liu, P., Zhang, J., Leung, C. W. K., He, C., and Griffiths, T. L. (2018). Exploiting Effective Representations for Chinese Sentiment Analysis using a Multi-channel Convolutional Neural Network. arXiv preprint arXiv:1808.02961.
* Song, Y., Shi, S., Li, J., and Zhang, H. (2018). Directional Skip-gram: Explicitly Distinguishing Left and Right Context for Word Embeddings. *NAACL-HLT 2018*.
* Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.
* Sun, C., Qiu, X., Xu, Y., and Huang, X. (2019). How to Fine-tune BERT for Text Classification? *CCL 2019*.
* Cui, Y., Che, W., Liu, T., Qin, B., Yang, Z., Wang, S., and Hu, G. (2019). Pre-training with Whole Word Masking for Chinese BERT. *EMNLP 2020*.
