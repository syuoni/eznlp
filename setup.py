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
      long_description=readme,
      url='https://github.com/syuoni/eznlp',
      author='Enwei Zhu',
      author_email='enwei.zhu@outlook.com',
      license='MIT',
      keywords='torch',
      packages=find_packages(include=["eznlp", "eznlp.*"]),
      install_requires=["torch>=1.7.1",
                        "torchtext==0.8.1",
                        "transformers==4.3.2",
                        "tokenizers==0.10.1", 
                        "allennlp==2.0.1", 
                        "flair==0.7",
                        "truecase==0.0.12", 
                        "hanziconv==0.3.2", 
                        "spacy>=2.3.2",
                        "jieba>=0.42.1", 
                        "numpy>=1.18.5",
                        "pandas>=1.0.5", 
                        "matplotlib>=3.2.2"],
      package_data={'eznlp': ["utils/transition.xlsx"]}, 
      include_package_data=True, 
      python_requires ='>=3.8,<4')

