#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import time

from base.logging import *
from base.util import *
from spec.parsed_spec import ParsedSpec

from bayesian.sampler import DefaultSampler
from bayesian.strategy import BayesianStrategy, BayesianPortfolioStrategy


strategies = {
  'bayesian': lambda sampler, params: BayesianStrategy(sampler, **params),
  'portfolio': lambda sampler, params: BayesianPortfolioStrategy(sampler, **params),
}


class HyperTuner(object):
  def __init__(self, hyper_params_spec, solver_generator, **strategy_params):
    self.solver_generator = solver_generator

    self.parsed = ParsedSpec(hyper_params_spec)
    info('Spec size=%d' % self.parsed.size())

    sampler = DefaultSampler()
    sampler.add_uniform(self.parsed.size())

    strategy_gen = as_function(strategy_params.get('strategy', 'bayesian'), presets=strategies)
    self.strategy = strategy_gen(sampler, strategy_params)

    self.timeout = strategy_params.get('timeout', 10)

  def tune(self):
    info('Start hyper-tuner')

    while True:
      point = self.strategy.next_proposal()
      hyper_params = self.parsed.instantiate(point)
      solver = self.solver_generator(hyper_params)

      tf_reset_all()
      accuracy = solver.train()

      previous_max = max(self.strategy.values) if len(self.strategy.values) > 0 else 0
      self.strategy.add_point(point, accuracy)
      index = len(self.strategy.values)

      marker = '!' if accuracy > previous_max else ' '
      info('%s [%d] accuracy=%.4f, params: %s' % (marker, index, accuracy, dict_to_str(hyper_params)))
      info('Current top-%d:' % min(len(self.strategy.values), 5))
      for value in sorted(self.strategy.values, reverse=True)[:5]:
        info('  accuracy=%.4f' % value)

      if self.timeout:
        time.sleep(self.timeout)


def tf_reset_all():
  import tensorflow as tf
  tf.reset_default_graph()
