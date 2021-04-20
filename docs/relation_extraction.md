## Relation Extraction (English)
Configurations of our implementation:
* Word embeddings are initialized with GloVe
* `From-scratch` models
    * Optimizer: Adadelta (lr=1.0)
    * Batch size: 64
    * Number of epochs: 100
* `Fine-tuining` models
    * Optimizer: AdamW (lr=1e-3/2e-3, ft_lr=1e-4)
    * Batch size: 48
    * Number of epochs: 50
    * Scheduler: Learning rate warmup at the first 20% steps followed by linear decay
    * BERT/RoBERTa models are loaded with dropout rate of 0.2

### CoNLL 2004 
| Model | Paper | Reported F1 | Our Imp. F1 (Pipline) | Our Imp. F1 (Joint) | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-------------------:|:---:|
| SpERT (with CharLSTM + LSTM)| -                     | -             | 86.92 / |  | num_layers=2 |
| SpERT (with BERT-base)    | Eberts and Ulges (2019) | 88.94 / 71.47 | 89.50 / |  | 
| SpERT (with BERT-base + LSTM) | -                   | -             | 90.21 / |  | 
| SpERT (with RoBERTa-base)        | -                | -             | 89.91 / |  | 
| SpERT (with RoBERTa-base + LSTM) | -                | -             | 90.30 / |  | 


### SciERC


### ADE



## References
* Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. 
* Eberts, M., and Ulges, A. (2019). Span-based joint entity and relation extraction with Transformer pre-training. *ECAI 2020*.
