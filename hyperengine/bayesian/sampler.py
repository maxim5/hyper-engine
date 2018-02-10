#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np
from six.moves import xrange

class BaseSampler(object):
  """
  Represents a helper class that allows to sample from high-dimensional spaces, usually unit hypercube.
  """
  def sample(self, size):
    """
    Returns a sample of the specified `size` (numpy array).
    Note that `size` defines just one dimension of the result.
    Other dimensions (usually hypercube) are fixed for the particular sampler instance.
    """
    raise NotImplementedError()


class DefaultSampler(BaseSampler):
  """
  Represents the default implementation of the sampler.
  """

  def __init__(self):
    self._random_functions = []

  def add(self, func):
    assert callable(func)
    self._random_functions.append(func)

  def add_uniform(self, size):
    func = lambda : np.random.uniform(0, 1, size=(size,))
    self._random_functions.append(func)

  def sample(self, size):
    result = []
    for i in xrange(size):
      item = [func() for func in self._random_functions]
      result.append(np.array(item).flatten())
    return np.array(result)
