#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import os
from setuptools import setup, find_packages

def read(file_):
  return open(os.path.join(os.path.dirname(__file__), file_)).read()

setup(
  name = 'hyper-engine',
  version = '0.1.0',
  author = 'Maxim Podkolzine',
  author_email = 'maxim.podkolzine@gmail.com',
  description = 'Python library for hyper-parameters optimization',
  license = 'Apache 2.0',
  keywords = 'machine learning, hyper-parameters, model selection, bayesian optimization',
  url = 'https://github.com/maxim5/hyper-engine',
  packages=find_packages(exclude=['tests']),
  long_description=read('README.rst'),
)
