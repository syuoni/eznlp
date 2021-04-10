## Text Classification (English)
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

### IMDb
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | -                    | -     | 91.58 | num_layers=1 |
| LSTM + Attention         | McCann et al. (2017) | 91.1  | 92.09 | num_layers=1 |
| BERT-base + Attention    | Sun et al. (2019)    | 94.60 | 
| RoBERTa-base + Attention | -                    | -     | 

### Yelp Full
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | Zhang et al. (2015)  | 58.17 | 65.97 | num_layers=2 |
| LSTM + Attention         | -                    | -     | 68.61 | num_layers=2 |
| BERT-base + Attention    | Sun et al. (2019)    | 69.94 | 
| RoBERTa-base + Attention | -                    | -     | 

### Yelp 2013 (with User and Product IDs)
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | Chen et al. (2016)   | 62.7  | 64.96 | num_layers=2 |
| LSTM + Attention         | Chen et al. (2016)   | 63.1  | 64.84 | num_layers=2 |
| BERT-base + Attention    | -                    | -     | 68.76 |
| RoBERTa-base + Attention | -                    | -     | 70.80 |


## Text Classification (Chinese)
Configurations of our implementation:
* Word-based (tokenized by `jieba`)

### ChnSentiCorp
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | - |  | 
| LSTM + Attention         | - |  | 
| BERT-base + Attention    | - |  | 
| RoBERTa-base + Attention | - |  | 

### THUCNews-10
| Model | Paper | Reported Acc. | Our Implementation Acc. | Notes |
|:-----:|:-----:|:-------------:|:-----------------------:|:-----:|
| LSTM + MaxPooling        | - |  | 
| LSTM + Attention         | - |  | 
| BERT-base + Attention    | - |  | 
| RoBERTa-base + Attention | - |  | 


## References
* Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level convolutional networks for text classification. *NIPS 2015*.
* Chen, H., Sun, M., Tu, C., Lin, Y., and Liu, Z. (2016). Neural sentiment classification with user and product attention. *EMNLP 2016*.
* McCann, B., Bradbury, J., Xiong, C., and Socher, R. (2017). Learned in translation: Contextualized word vectors. *NIPS 2017*.
* Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. 
* Sun, C., Qiu, X., Xu, Y., and Huang, X. (2019). How to fine-tune BERT for text classification? *CCL 2019*.
