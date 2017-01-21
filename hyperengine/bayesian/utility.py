#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import math
import numpy as np
from scipy import stats


class BaseUtility(object):
  """
  Utility is a function that evaluates a potential of the points in high-dimensional spaces.
  Utility can use prior information about the true values over certain points to model the target function,
  thus predict the points that are likely to be the maximum.
  """

  def __init__(self, points, values, **params):
    super(BaseUtility, self).__init__()
    self.points = np.array(points)
    self.values = np.array(values)

    if len(self.points.shape) == 1:
      self.points = self.points.reshape(-1, 1)
    assert len(self.points.shape) == 2
    assert len(self.values.shape) == 1
    assert self.points.shape[0] == self.values.shape[0]

    self.dimension = self.points.shape[1]
    self.iteration = params.get('iteration', self.points.shape[0])

  def compute_values(self, batch):
    """
    Evaluates the utility function for a batch of points. Returns the numpy array.
    """
    raise NotImplementedError()


class BaseGaussianUtility(BaseUtility):
  """
  Represents the utility function based on Gaussian Process models.
  """

  def __init__(self, points, values, kernel, mu_prior=0, noise_sigma=0.0, **params):
    super(BaseGaussianUtility, self).__init__(points, values, **params)
    self.kernel = kernel

    mu_prior = np.array(mu_prior)
    if len(mu_prior.shape) == 0:
      mu_prior_values, mu_prior_star = mu_prior, mu_prior
    else:
      mu_prior_values, mu_prior_star = mu_prior[:-1], mu_prior[-1]

    kernel_matrix = self.kernel.compute(self.points) + np.eye(self.points.shape[0]) * noise_sigma**2
    self.k_inv = np.linalg.pinv(kernel_matrix)
    self.k_inv_f = np.dot(self.k_inv, (self.values - mu_prior_values))
    self.mu_prior_star = mu_prior_star

  def mean_and_std(self, batch):
    assert len(batch.shape) == 2

    batch = np.array(batch)
    k_star = np.swapaxes(self.kernel.compute(self.points, batch), 0, 1)
    k_star_star = self.kernel.id(batch)

    mu_star = self.mu_prior_star + np.dot(k_star, self.k_inv_f)

    t_star = np.dot(self.k_inv, k_star.T)
    t_star = np.einsum('ij,ji->i', k_star, t_star)
    sigma_star = k_star_star - t_star

    return mu_star, sigma_star


class ProbabilityOfImprovement(BaseGaussianUtility):
  """
  Implements the PI method.
  See the following sources for more details:
  H. J. Kushner. A new method of locating the maximum of an arbitrary multipeak curve in the presence of noise.
  J. Basic Engineering, 86:97–106, 1964.
  """

  def __init__(self, points, values, kernel, mu_prior=0, noise_sigma=0.0, **params):
    super(ProbabilityOfImprovement, self).__init__(points, values, kernel, mu_prior, noise_sigma, **params)
    self.epsilon = params.get('epsilon', 1e-8)
    self.max_value = np.max(self.values)

  def compute_values(self, batch):
    mu, sigma = self.mean_and_std(batch)
    z = (mu - self.max_value - self.epsilon) / sigma
    cdf = stats.norm.cdf(z)
    cdf[np.abs(sigma) < self.epsilon] = 0.0
    return cdf


class ExpectedImprovement(BaseGaussianUtility):
  """
  Implements the EI method.
  See the following sources for more details:
  J. Mockus, V. Tiesis, and A. Zilinskas. Toward Global Optimization, volume 2,
  chapter The Application of Bayesian Methods for Seeking the Extremum, pages 117–128. Elsevier, 1978.
  """

  def __init__(self, points, values, kernel, mu_prior=0, noise_sigma=0.0, **params):
    super(ExpectedImprovement, self).__init__(points, values, kernel, mu_prior, noise_sigma, **params)
    self.epsilon = params.get('epsilon', 1e-8)
    self.max_value = np.max(self.values)

  def compute_values(self, batch):
    mu, sigma = self.mean_and_std(batch)
    z = (mu - self.max_value - self.epsilon) / sigma
    ei = (mu - self.max_value - self.epsilon) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    ei[np.abs(sigma) < self.epsilon] = 0.0
    return ei


class UpperConfidenceBound(BaseGaussianUtility):
  """
  Implements the UCB method.
  See the following sources for more details:
  Peter Auer, Using Confidence Bounds for Exploitation-Exploration Trade-offs,
  Journal of Machine Learning Research 3 (2002) 397-422, 2011.
  """

  def __init__(self, points, values, kernel, mu_prior=0, noise_sigma=0.0, **params):
    super(UpperConfidenceBound, self).__init__(points, values, kernel, mu_prior, noise_sigma, **params)
    delta = params.get('delta', 0.5)
    self.beta = np.sqrt(2 * np.log(self.dimension * self.iteration**2 * math.pi**2 / (6 * delta)))

  def compute_values(self, batch):
    mu, sigma = self.mean_and_std(batch)
    return mu + self.beta * sigma


class RandomPoint(BaseUtility):
  """
  A naive random point method. All points are picked equally likely, thus utility method is constant everywhere.
  """

  def __init__(self, points, values, **params):
    super(RandomPoint, self).__init__(points, values, **params)

  def compute_values(self, batch):
    return np.zeros(len(batch))
