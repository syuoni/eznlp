# Easy Natural Language Processing

This repository consists of:
* `eznlp.text_classification`
* `eznlp.sequence_tagging`
* `eznlp.language_modeling`

## Experiments
### Named Entity Recognition
#### CoNLL 2003 (English)

| Model    | Paper | Reported F1 | Our Implementation | Optimization Notes | Other Notes |
|:--------:|:-----:|:-----------:|:------------------:|:------------------:|:-----------:|
| CharLSTM + LSTM + CRF       | Lample et al. (2016) | 90.94         | 91.28 | SGD (lr=0.1); batch_size=32 | num_layers=1 |
| CharCNN + LSTM + CRF        | Ma and Hovy (2016)   | 91.21         | 90.70 | SGD (lr=0.1); batch_size=32 | num_layers=1 |
| ELMo + Char + LSTM + CRF    | Peters et al. (2018) | 92.22 (0.10)  | 92.76 | SGD (lr=0.1); batch_size=16 | num_layers=1 |
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)  | 93.09* (0.12) | 92.60 | SGD (lr=0.1); batch_size=32 | num_layers=1 |
| ELMo + Flair + Char + LSTM + CRF | -               | -             | 92.67 | SGD (lr=0.1); batch_size=32 | num_layers=1 |
| BERT-base + Softmax         | Devlin et al. (2018) | 92.4          | 92.19 | AdamW (lr=1e-3, ft_lr=1e-5); warmup; batch_size=64 | bert_drop=0.2; truecase |
| BERT-base + CRF             | -                    | -             | 92.33 | AdamW (lr=1e-3, ft_lr=1e-5); warmup; batch_size=64 | bert_drop=0.2; truecase |
| BERT-base + LSTM + CRF      | -                    | -             |  |
| BERT-large + Softmax        | Devlin et al. (2018) | 92.8          |  |
| BERT-large + CRF            | -                    | -             |  |
| BERT-large + LSTM + CRF     | -                    | -             |  |
| RoBERTa-base + Softmax      | Liu et al. (2019)    | -             | 92.61 | AdamW (lr=1e-3, ft_lr=1e-5); warmup; batch_size=64 | bert_drop=0.2 |
| RoBERTa-base + CRF          | -                    | -             | 92.69 | AdamW (lr=1e-3, ft_lr=1e-5); warmup; batch_size=64 | bert_drop=0.2 |
| RoBERTa-base + LSTM + CRF   | -                    | -             |  |
| RoBERTa-large + Softmax     | Liu et al. (2019)    | -             |  |
| RoBERTa-large + CRF         | -                    | -             |  |
| RoBERTa-large + LSTM + CRF  | -                    | -             |  |

\* Model trained on both training and development sets. 

#### OntoNotes v5 (English)
| Model    | Paper | Reported F1 | Our Implementation | Optimization Notes | Other Notes |
|:--------:|:-----:|:-----------:|:------------------:|:------------------:|:-----------:|
| CharLSTM + LSTM + CRF       | Lample et al. (2016)    | -            | 87.97 | SGD (lr=0.1); batch_size=10 | num_layers=2 |
| CharCNN + LSTM + CRF        | Chiu and Nichols (2016) | 86.17 (0.22) | 87.49 | SGD (lr=0.1); batch_size=10 | num_layers=2 |
| ELMo + Char + LSTM + CRF    | Peters et al. (2018)    | -            |       | SGD (lr=0.1); batch_size=XX | num_layers=2 |
| Flair + Char + LSTM + CRF   | Akbik et al. (2018)     | -            |       | SGD (lr=0.1); batch_size=XX | num_layers=2 |
| ELMo + Flair + Char + LSTM + CRF | -                  | -            |       | SGD (lr=0.1); batch_size=XX | num_layers=2 |
| BERT-base + Softmax         | Devlin et al. (2018)    |


#### SIGHAN 2006 / MSRA (Chinese)
| Model    | Paper | Reported F1 | Our Implementation | Optimization Notes | Other Notes |
|:--------:|:-----:|:-----------:|:------------------:|:------------------:|:-----------:|
| LSTM + CRF               | Zhang and Yang (2018) | 88.81 |
| Bichar + LSTM + CRF      | Zhang and Yang (2018) | 91.87 |
| Lattice-LSTM + CRF       | Zhang and Yang (2018) | 93.18 |
| SoftLexicon + LSTM + CRF | Ma et al. (2020)      | 93.66 |
| BERT + CRF               | Ma et al. (2020)      | 93.76 |
| BERT + LSTM + CRF        | Ma et al. (2020)      | 94.83 |
| SoftLexicon + BERT + CRF | Ma et al. (2020)      | 95.42 |

#### WeiboNER v2 (Chinese)

#### ResumeNER (Chinese)

#### OntoNotes v5 (Chinese)




## Future Plans
- [ ] SoftLexicon
- [ ] Experiments on Chinese NER datasets
- [ ] Experiments on text classification datasets
- [ ] Focal loss (and combined to CRF?)
- [ ] Relation Extraction
- [ ] Span-based models (e.g., SpERT)


## References
* Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. In *Proceedings of the 2016 Conference of the NAACL-HLT*.
* Ma, X., & Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. In *Proceedings of the 54th Annual Meeting of the ACL*.
* Chiu, J. P., & Nichols, E. (2016). Named Entity Recognition with Bidirectional LSTM-CNNs. *Transactions of the Association for Computational Linguistics*, 4:357-370.
* Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep Contextualized Word Representations. In *Proceedings of the 2018 Conference of the NAACL-HLT*.
* Akbik, A., Blythe, D., & Vollgraf, R. (2018). Contextual String Embeddings for Sequence Labeling. In *Proceedings of the 27th International Conference on Computational linguistics*.
* Zhang, Y., & Yang, J. (2018). Chinese NER Using Lattice LSTM. In *Proceedings of the 56th Annual Meeting of the ACL*.
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the NAACL-HLT*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. 
* Ma, R., Peng, M., Zhang, Q., Wei, Z., & Huang, X. J. (2020). Simplify the Usage of Lexicon in Chinese NER. In *Proceedings of the 58th Annual Meeting of the ACL*.
