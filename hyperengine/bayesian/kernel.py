#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


class BaseKernel(object):
  """
  Represents a kernel function See https://en.wikipedia.org/wiki/Positive-definite_kernel for details.
  All methods operate with a batch of points.
  """

  def id(self, batch_x):
    """
    Computes the kernel function of a point (a batch of points) with itself.
    Standard kernels satisfy `k(x, x) = 1`, but a derived kernel may compute it differently.
    Note that the result vector equals to the diagonal of `compute(batch_x)`
    """
    raise NotImplementedError()

  def compute(self, batch_x, batch_y=None):
    """
    Computes the kernel matrix, consisting of kernel values for each pair of points.
    If `batch_y is None`, the result is a square matrix computed from `batch_x`.
    If `batch_y is not None`, the result is computed from both `batch_x` and `batch_y`.
    """
    raise NotImplementedError()


class RadialBasisFunction(BaseKernel):
  """
  Implements RBF kernel function.
  See https://en.wikipedia.org/wiki/Radial_basis_function_kernel
  """

  def __init__(self, gamma=0.5, **params):
    self._gamma = gamma
    self._params = params
    self._params.setdefault('metric', 'sqeuclidean')

  def id(self, batch_x):
    return np.ones(batch_x.shape[:1])

  def compute(self, batch_x, batch_y=None):
    if len(batch_x.shape) == 1:
      batch_x = batch_x.reshape((1, -1))
    if batch_y is not None and len(batch_y.shape) == 1:
      batch_y = batch_y.reshape((1, -1))

    if batch_y is None:
      dist = squareform(pdist(batch_x, **self._params))
    else:
      dist = cdist(batch_x, batch_y, **self._params)
    return np.exp(-self._gamma * dist)
