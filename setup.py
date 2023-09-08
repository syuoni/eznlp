# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup, find_packages

HERE = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(HERE, "README.md"), encoding='utf-8') as f:
    readme = f.read()
with open(os.path.join(HERE, 'eznlp', '__init__.py'), encoding='utf-8') as f:
    version = re.search(r'__version__ = (["\'])([^"\']*)\1', f.read())[2]


setup(name='eznlp',
      version=version,
      description='Easy Natural Language Processing',
      long_description_content_type='text/markdown',
      long_description=readme,
      url='https://github.com/syuoni/eznlp',
      author='Enwei Zhu',
      author_email='enwei.zhu@outlook.com',
      license='Apache',
      keywords='torch',
      packages=find_packages(include=["eznlp", "eznlp.*"]),
      include_package_data=True,
      install_requires=["torch>=1.7.1",
                        "torchvision>=0.8.2",
                        "flair==0.8",
                        "allennlp==2.1.0",
                        "transformers==4.3.2",
                        "tokenizers==0.10.1",
                        "nltk>=3.5",
                        "truecase==0.0.12",
                        "hanziconv==0.3.2",
                        "spacy>=2.3.2",
                        "jieba>=0.42.1",
                        "numpy>=1.18.5",
                        "pandas>=1.0.5", 
                        "matplotlib>=3.2.2"],
      tests_require=["torchtext>=0.8.1",
                     "pytorch-crf>=0.7.2"], 
      python_requires='>=3.8,<4')
