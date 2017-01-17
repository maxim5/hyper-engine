#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


class BaseKernel(object):
  def id(self, batch_x):
    pass

  def compute(self, batch_x, batch_y=None):
    pass


class RadialBasisFunction(BaseKernel):
  def __init__(self, gamma=0.5, **params):
    self.gamma = gamma
    self.params = params
    self.params.setdefault('metric', 'sqeuclidean')

  def id(self, batch_x):
    return np.ones(batch_x.shape[:1])

  def compute(self, batch_x, batch_y=None):
    if len(batch_x.shape) == 1:
      batch_x = batch_x.reshape((1, -1))
    if batch_y is not None and len(batch_y.shape) == 1:
      batch_y = batch_y.reshape((1, -1))

    if batch_y is None:
      dist = squareform(pdist(batch_x, **self.params))
    else:
      dist = cdist(batch_x, batch_y, **self.params)
    return np.exp(-self.gamma * dist)
