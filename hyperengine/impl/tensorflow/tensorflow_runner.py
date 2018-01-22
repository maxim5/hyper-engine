#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import tensorflow as tf

from ...base import *
from hyperengine.model.base_runner import BaseRunner
from tf_util import graph_vars, get_total_dim


class TensorflowRunner(BaseRunner):
  def __init__(self, model=None, extra_feed={}):
    super(TensorflowRunner, self).__init__()

    self._graph = model or tf.get_default_graph()
    assert isinstance(self._graph, tf.Graph), '"model" argument must be either tf.Graph instance or None'
    self._extra_feed = extra_feed
    assert isinstance(self._extra_feed, dict), '"extra_feed" must be a dictionary (tensor -> value)'

  def build_model(self):
    self._x = self._find_tensor('input')
    self._y = self._find_tensor('label')
    self._mode = self._find_tensor('mode', mandatory=False)
    self._loss = self._find_tensor('loss')
    self._accuracy = self._find_tensor('accuracy')
    self._minimize = self._find_op('minimize')
    self._model_size = self._calc_model_size()

  def init(self, **kwargs):
    self._session = kwargs['session']
    init = self._find_op('initializer', tf.global_variables_initializer())
    self._session.run(init)

  def run_batch(self, batch_x, batch_y):
    feed_dict = self._get_feed_dict(batch_x, batch_y, 'train')
    self._session.run(self._minimize, feed_dict=feed_dict)

  def evaluate(self, batch_x, batch_y):
    feed_dict = self._get_feed_dict(batch_x, batch_y, 'test')
    loss, accuracy = self._session.run([self._loss, self._accuracy], feed_dict=feed_dict)
    return {'loss': loss, 'accuracy': accuracy}

  def terminate(self):
    tf.reset_default_graph()

  def graph(self):
    return self._graph

  def model_size(self):
    return self._model_size

  def _get_feed_dict(self, batch_x, batch_y, mode):
    feed_dict = {self._x: batch_x, self._y: batch_y}
    if self._mode is not None:
      feed_dict[self._mode] = mode
    for k, v in self._extra_feed.items():
      if isinstance(v, dict) and mode in v.keys():
        v = v[mode]
      feed_dict[k] = v
    return feed_dict

  def _find_tensor(self, name, mandatory=True):
    try:
      return self._graph.get_tensor_by_name(name + ':0')
    except KeyError:
      if not mandatory:
        return None
      warn('Failed to infer a tensor "%s" in Tensorflow graph. '
           'Most likely, you should add "name=\'%s\'" in the placeholder/tensor definition' % (name, name))
      raise

  def _find_op(self, name, default=None):
    try:
      return self._graph.get_operation_by_name(name)
    except KeyError:
      if default is not None:
        return default
      warn('Failed to infer an op "%s" in Tensorflow graph. '
           'Most likely, you should add "name=\'%s\'" in the op definition' % (name, name))
      raise

  def _calc_model_size(self):
    return sum(get_total_dim(element) for element in graph_vars(self._graph))
