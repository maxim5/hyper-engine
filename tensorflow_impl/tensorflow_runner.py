#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


from model.base_runner import BaseRunner
from base.logging import *
from base.util import *


class TensorflowRunner(BaseRunner):
  def __init__(self, model):
    super(TensorflowRunner, self).__init__()
    self._model = model
    self._session = None

  def build_model(self, **kwargs):
    self._session = kwargs['session']
    functions = self._model.build_graph()
    assert len(functions) >= 4
    self._init, self._optimizer, self._cost, self._accuracy = functions[:4]
    self._additional = functions[4:]
    self._session.run(self._init, feed_dict=self._model.feed_dict(mode='train'))

    info('Start training. Model size: %dk' % (self._model.params_num() / 1000))
    info('Hyper params: %s' % dict_to_str(self._model.hyper_params()))

  def run_batch(self, batch_x, batch_y):
    self._session.run(self._optimizer, feed_dict=self._model.feed_dict(x=batch_x, y=batch_y, mode='train'))

  def evaluate(self, batch_x, batch_y):
    fetches = [self._cost, self._accuracy] + self._additional
    result = self._session.run(fetches, feed_dict=self._model.feed_dict(x=batch_x, y=batch_y, mode='test'))
    assert len(result) >= 2
    cost, accuracy = result[:2]
    data = result[2:]
    return {'cost': cost, 'accuracy': accuracy, 'data': data}

  def describe(self):
    return {'model_size': self._model.params_num(), 'hyper_params': self._model.hyper_params()}
