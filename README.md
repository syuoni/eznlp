# Easy Natural Language Processing

Overparameterized neural networks are lazy (Chizat et al., 2019), so we design structures and objectives that can be easily optimized. 

`eznlp` is a `PyTorch`-based package for neural natural language processing, currently supporting the following tasks:

* Text Classification ([Experimental Results](docs/text-classification.md))
* Named Entity Recognition ([Experimental Results](docs/entity-recognition.md))
* Relation Extraction ([Experimental Results](docs/relation-extraction.md))
* Attribute Extraction
* Machine Translation
* Image Captioning

This repository also maintains the code of our papers: 
* Check this [link](docs/deep-span.md) for "Deep Span Representations for Named Entity Recognition" accepted to *Findings of ACL 2023*. 
* Check this [link](docs/boundary-smoothing.md) for "Boundary Smoothing for Named Entity Recognition" in *ACL 2022*. 
* Check the [annotation scheme](publications/framework/scheme.pdf) and [HwaMei-500](publications/framework/HwaMei-500.md) dataset described in "A Unified Framework of Medical Information Annotation and Extraction for Chinese Clinical Text" on *Artificial Intelligence in Medicine*. 


## Installation
### Create an environment
```bash
$ conda create --name eznlp python=3.8
$ conda activate eznlp
```

### Install dependencies
```bash
$ conda install numpy=1.18.5 pandas=1.0.5 xlrd=1.2.0 matplotlib=3.2.2 
$ conda install pytorch=1.7.1 torchvision=0.8.2 torchtext=0.8.1 {cpuonly|cudatoolkit=10.2|cudatoolkit=11.0} -c pytorch 
$ pip install -r requirements.txt 
```

### Install `eznlp`
* From source (recommended)
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
@inproceedings{zhu2023deep,
  title={Deep Span Representations for Named Entity Recognition},
  author={Zhu, Enwei and Liu, Yiyang and Li, Jinpeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  month={jul},
  year={2023},
  address={Toronto, Canada},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/2023.findings-acl.672},
  doi={10.18653/v1/2023.findings-acl.672},
  pages={10565--10582}
}
```

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
  doi={10.18653/v1/2022.acl-long.490},
  pages={7096--7108}
}
```

```
@article{zhu2023framework,
  title={A unified framework of medical information annotation and extraction for {C}hinese clinical text},
  author={Zhu, Enwei and Sheng, Qilin and Yang, Huanwan and Liu, Yiyang and Cai, Ting and Li, Jinpeng},
  journal={Artificial Intelligence in Medicine},
  volume={142},
  pages={102573},
  year={2023},
  publisher={Elsevier}
}
```


## References
* Chizat, L., Oyallon, E., and Bach, F. On lazy training in differentiable programming. In *NeurIPS 2019*. 
