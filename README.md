# Easy Natural Language Processing

`eznlp` is a `PyTorch`-based package for neural natural language processing, currently supporting the following tasks:

* Text Classification ([Experimental Results](docs/text-classification.pdf))
* Named Entity Recognition ([Experimental Results](docs/entity-recognition.pdf))
    * Sequence Tagging
    * Span Classification
    * Boundary Selection
* Relation Extraction ([Experimental Results](docs/relation-extraction.pdf))
* Attribute Extraction
* Machine Translation
* Image Captioning

This repository also maintains the code of our papers: 
* Check this [link](docs/boundary-smoothing.md) for "Boundary Smoothing for Named Entity Recognition" accepted to ACL 2022 main conference. 


## Installation
### With `pip`
```bash
$ pip install eznlp
```

### From source
```bash
$ python setup.py sdist
$ pip install dist/eznlp-<version>.tar.gz
```


## Running the Code
### Text classification
```bash
$ python scripts/text_classification.py --dataset <dataset> [options]
```

### Entity recognition
```bash
$ python scripts/entity_recognition.py --dataset <dataset> [options]
```

### Relation extraction
```bash
$ python scripts/relation_extraction.py --dataset <dataset> [options]
```

### Attribute extraction
```bash
$ python scripts/attribute_extraction.py --dataset <dataset> [options]
```


## Citation
If you find our code useful, please cite the following papers: 

```
@inproceedings{zhu2022boundary,
  title={Boundary Smoothing for Named Entity Recognition},
  author={Zhu, Enwei and Li, Jinpeng},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022},
  publisher={Association for Computational Linguistics},
}
```

```
@article{zhu2021framework,
  title={A Unified Framework of Medical Information Annotation and Extraction for {Chinese} Clinical Text},
  author={Zhu, Enwei and Sheng, Qilin and Yang, Huanwan and Li, Jinpeng},
  journal={arXiv preprint arXiv:2203.03823},
  year={2021}
}
```


## Future Plans
- [ ] Unify the data interchange format as a dict, i.e., `entry`
- [ ] Reorganize `JsonIO`
- [ ] Memory optimization for large dataset for training PLM
- [ ] More relation extraction models
- [ ] Multihot classification
- [ ] Unify the aggregation interface of pooling and attention
- [ ] Radical-level features
- [ ] Data augmentation
- [ ] Loss increases in later training phases -> LR finder?
