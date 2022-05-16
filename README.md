# Easy Natural Language Processing

Overparameterized neural networks are lazy (Chizat et al., 2019), so we design structures and objectives that can be easily optimized. 

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
* Check this [link](publications/framework/scheme.pdf) for the annotation scheme described in "A Unified Framework of Medical Information Annotation and Extraction for Chinese Clinical Text". 


## Installation
### Install dependencies
```bash
$ conda install numpy=1.18.5 pandas=1.0.5 xlrd=1.2.0 matplotlib=3.2.2 
$ conda install pytorch=1.7.1 torchvision=0.8.2 torchtext=0.8.1 {cpuonly|cudatoolkit=10.2} -c pytorch 
$ pip install -r requirements.txt 
```

### Install `eznlp`
* From source (suggested)
```bash
$ python setup.py sdist
$ pip install dist/eznlp-<version>.tar.gz --no-deps
```

* With `pip`
```bash
$ pip install eznlp --no-deps
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
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month={may},
  year={2022},
  address={Dublin, Ireland},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/2022.acl-long.490},
  pages={7096--7108}
}
```

```
@article{zhu2021framework,
  title={A Unified Framework of Medical Information Annotation and Extraction for {C}hinese Clinical Text},
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


## References
* Chizat, L., Oyallon, E., and Bach, F. (2019). On lazy training in differentiable programming. *NeurIPS 2019*, 2937–2947. 
