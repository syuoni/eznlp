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
      python_requires='>=3.8,<4')
