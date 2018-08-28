#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np
import time

from ..base import *
from ..spec import ParsedSpec
from ..bayesian.sampler import DefaultSampler
from ..bayesian.strategy import BayesianStrategy, BayesianPortfolioStrategy


strategies = {
  'bayesian': lambda sampler, params: BayesianStrategy(sampler, **params),
  'portfolio': lambda sampler, params: BayesianPortfolioStrategy(sampler, **params),
}


class HyperTuner(object):
  def __init__(self, hyper_params_spec, solver_generator, **strategy_params):
    self._solver_generator = solver_generator

    self._parsed = ParsedSpec(hyper_params_spec)
    info('Spec size:', self._parsed.size())

    sampler = DefaultSampler()
    sampler.add_uniform(self._parsed.size())

    strategy_gen = as_function(strategy_params.get('strategy', 'bayesian'), presets=strategies)
    self._strategy = strategy_gen(sampler, strategy_params)

    self._timeout = strategy_params.get('timeout', 0)
    self._max_points = strategy_params.get('max_points', None)

  def tune(self):
    info('Start hyper tuner')

    while True:
      if self._max_points is not None and len(self._strategy.values) >= self._max_points:
        info('Maximum points reached: max_points=%d. Aborting hyper tuner' % self._max_points)
        break

      point = self._strategy.next_proposal()
      hyper_params = self._parsed.instantiate(point)
      solver = self._solver_generator(hyper_params)

      accuracy = solver.train()

      previous_max = np.max(self._strategy.values) if len(self._strategy.values) > 0 else -np.inf
      self._strategy.add_point(point, accuracy)
      index = len(self._strategy.values)

      marker = '!' if accuracy > previous_max else ' '
      info('%s [%d] accuracy=%.4f, params: %s' % (marker, index, accuracy, smart_str(hyper_params)))
      info('Current top-%d:' % min(len(self._strategy.values), 5))
      for value in sorted(self._strategy.values, reverse=True)[:5]:
        info('  accuracy=%.4f' % value)

      if self._timeout:
        time.sleep(self._timeout)

      solver.terminate()
