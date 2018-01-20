#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np

from ..base import *


class BaseUtilityMaximizer(object):
  """
  Represents a helper class that computes the max value of an utility method
  over a certain domain, usually unit N-dimensional hypercube.
  """
  def __init__(self, utility):
    super(BaseUtilityMaximizer, self).__init__()
    self._utility = utility

  def compute_max_point(self):
    """
    Returns the point that reaches the maximum value.
    """
    raise NotImplementedError()


class MonteCarloUtilityMaximizer(BaseUtilityMaximizer):
  """
  A simple implementation that applies Monte-Carlo method to pick the maximum.
  """

  def __init__(self, utility, sampler, **params):
    super(MonteCarloUtilityMaximizer, self).__init__(utility)
    self._sampler = sampler
    self._batch_size = params.get('batch_size', 100000)
    self._tweak_probability = params.get('tweak_probability', 0)
    self._flip_probability = params.get('flip_probability', 0)

  def compute_max_point(self):
    batch = self._sampler.sample(size=self._batch_size)
    values = self._utility.compute_values(batch)
    i = np.argmax(values)
    debug('Max utility point:', batch[i])
    vlog('Max utility value:', values[i])
    point = self._tweak_randomly(batch[i], batch[0])
    return point

  def _tweak_randomly(self, optimal, point):
    if np.random.uniform() < self._tweak_probability:
      debug('Tweaking optimal point with probability:', self._flip_probability)
      p = self._flip_probability or min(2.0 / optimal.shape[0], 0.5)
      mask = np.random.choice([False, True], size=optimal.shape, p=[1-p, p])
      optimal[mask] = point[mask]
      debug('Selected point after random tweaking:', optimal)
    return optimal
