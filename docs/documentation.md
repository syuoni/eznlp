## Documentation

### Design of `Decoder` classes
|       | Methods in `eznlp.model.decoder.base` | Methods in Subclasses |
|:-----:|:-------------------------------------:|:---------------------:|
| `DecoderMixin`  | `_unsqueezed_retrieve`, <br> `_unsqueezed_evaluate` | `retrieve`, `evaluate`, <br> `exemplify`, `batchify` |
| `DecoderConfig` | `criterion`, `instantiate_criterion` | `build_vocab`, `instantiate` |
| `Decoder`       | `_unsqueezed_decode` | `forward`, `decode` |

### Design of `Model` classes
|       | Methods in `eznlp.model.model.base` | Methods in Subclasses |
|:-----:|:-----------------------------------:|:---------------------:|
| `ModelConfig` | `valid`, `name` | `exemplify`, `batchify`, <br> `build_vocabs_and_dims`, `instantiate` |
| `Model`       | `forward`, `decode` | `pretrained_parameters`, `forward2states` |
