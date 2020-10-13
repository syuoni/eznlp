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
      description='Natural Language Processing by Enwei Zhu',
      long_description=readme,
      url='https://github.com/syuoni/eznlp',
      author='Enwei Zhu',
      author_email='enwei.zhu@outlook.com',
      license='MIT',
      keywords='torch',
      packages=find_packages(include=["eznlp", "eznlp.*"]),
      install_requires=["torch>=1.6.0",
                        "torchtext>=0.7.0",
                        "pytorch-crf>=0.7.2", 
                        "transformers>=3.0.2",
                        "spacy>=2.3.2",
                        "numpy>=1.18.5",
                        "pandas>=1.0.5",
                        "tqdm",
                        "pytest"],
      package_data={'eznlp': ["sequence_tagging/transitions.xlsx"]}, 
      include_package_data=True, 
      python_requires ='>=3.8,<4')

