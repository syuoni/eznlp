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

♦ use both training and development splits for training ([SpERT](https://github.com/lavis-nlp/spert/issues/2#issuecomment-559775207)).  
♠️ do not consider entity type correctness when evaluating relation extraction.  

### CoNLL 2004 
| Model | Paper | Reported F1 | Our Imp. F1 (Pipeline) | Our Imp. F1 (Joint) | Notes |
|:-----:|:-----:|:-----------:|:---------------------:|:-------------------:|:-----:|
| SpERT (with CharLSTM + LSTM)| -                     | -               |  -            | 86.57 / 66.01 | num_layers=2 |
| SpERT (with BERT-base)    | Eberts and Ulges (2019) | 88.94♦ / 71.47♦ | 88.80 / 69.78 | 88.93 / 70.82 | 
| SpERT (with BERT-base + LSTM) | -                   | -               | 89.89 / 69.68 | 89.86 / 72.51 | 
| SpERT (with RoBERTa-base)        | -                | -               | 90.30 / 72.18 | 90.18 / 72.64 | 
| SpERT (with RoBERTa-base + LSTM) | -                | -               | 90.10 / 73.46 | 89.17 / 75.03 | 

### SciERC
| Model | Paper | Reported F1 | Our Imp. F1 (Joint) | Notes |
|:-----:|:-----:|:-----------:|:-------------------:|:-----:|
| SpERT (with CharLSTM + LSTM)| -                     | -                | 59.63 / 23.04 / 34.25♠️ | num_layers=2 |
| SpERT (with BERT-base)    | Eberts and Ulges (2019) | 67.62♦ / 46.44♦♠️ | 66.71 / 33.94 / 46.07♠️ | 
| SpERT (with BERT-base + LSTM) | -                   | -                | 67.47 / 33.67 / 45.82♠️ | 
| SpERT (with RoBERTa-base)        | -                | -                | 69.29 / 36.65 / 48.93♠️ | 
| SpERT (with RoBERTa-base + LSTM) | -                | -                | 68.89 / 34.65 / 47.52♠️ | 


## References
* Bekoulis, G., Deleu, J., Demeester, T., and Develder, C. (2018). Joint Entity Recognition and Relation Extraction as a Multi-head Selection Problem. *Expert Systems with Applications*, 114: 34-45.
* Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. 
* Eberts, M., and Ulges, A. (2019). Span-based Joint Entity and Relation Extraction with Transformer Pre-training. *ECAI 2020*.
