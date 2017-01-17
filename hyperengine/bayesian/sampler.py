#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np


class BaseSampler(object):
  def sample(self, size):
    pass


class DefaultSampler(BaseSampler):
  def __init__(self):
    self.random_functions = []

  def add(self, func):
    assert callable(func)
    self.random_functions.append(func)

  def add_uniform(self, size):
    func = lambda : np.random.uniform(0, 1, size=(size,))
    self.random_functions.append(func)

  def sample(self, size):
    result = []
    for i in xrange(size):
      item = [func() for func in self.random_functions]
      result.append(np.array(item).flatten())
    return np.array(result)
