# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup, find_packages

here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, "README.md")) as f:
    readme = f.read()
with open(os.path.join(here, 'eznlp', '__init__.py')) as f:
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
      install_requires=["torch==1.7.1",
                        "torchtext==0.8.1",
                        "allennlp==1.1.0", 
                        "transformers==3.0.2",
                        "flair==0.6.1",
                        "truecase==0.0.12", 
                        "spacy>=2.3.2",
                        "jieba>=0.42.1", 
                        "numpy>=1.18.5",
                        "pandas>=1.0.5",
                        "tqdm",
                        "pytest"],
      package_data={'eznlp': ["sequence_tagging/transition.xlsx"]}, 
      include_package_data=True, 
      python_requires ='>=3.8,<4')

