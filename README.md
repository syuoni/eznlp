# Easy Natural Language Processing

Overparameterized neural networks are often described as lazy (Chizat et al., 2019), which motivates us to design architectures and objectives that are easier to optimize.

`eznlp` is a `PyTorch`-based package for neural natural language processing, currently supporting the following tasks:

* Text Classification ([Experimental Results](docs/text-classification.md))
* Named Entity Recognition ([Experimental Results](docs/entity-recognition.md))
* Relation Extraction ([Experimental Results](docs/relation-extraction.md))
* Attribute Extraction
* Machine Translation
* Image Captioning

This repository also contains code for our published papers:
* See [this link](docs/deep-span.md) for *Deep Span Representations for Named Entity Recognition*, presented at *Findings of ACL 2023*.
* See [this link](docs/boundary-smoothing.md) for *Boundary Smoothing for Named Entity Recognition*, presented at *ACL 2022*.
* See the [annotation scheme](publications/framework/scheme.pdf) and [HwaMei-500](publications/framework/HwaMei-500.md) dataset described in *A Unified Framework of Medical Information Annotation and Extraction for Chinese Clinical Text* published in *Artificial Intelligence in Medicine*.

## Installation

### Create an Environment

We recommend using Docker. The latest tested image is `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`.
```bash
$ docker run --rm -it --gpus=all --mount type=bind,source=${PWD},target=/workspace/eznlp --workdir /workspace/eznlp pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
```

Alternatively, you can create a virtual environment. For example:
```bash
$ conda create --name eznlp python=3.11
$ conda activate eznlp
```

### Install `eznlp`

If you wish to use `eznlp` as a package, install it from PyPI:
```bash
$ pip install eznlp
```

If you plan to develop on this project, install it in editable mode:
```bash
$ pip install -e .
```

## Running the Code

### Text Classification
```bash
$ python scripts/text_classification.py --dataset <dataset> [options]
```

### Entity Recognition
```bash
$ python scripts/entity_recognition.py --dataset <dataset> [options]
```

### Relation Extraction
```bash
$ python scripts/relation_extraction.py --dataset <dataset> [options]
```

### Attribute Extraction
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
