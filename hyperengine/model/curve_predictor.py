#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np

from base_io import DefaultIO, Serializable
from base.logging import *
from hyperengine.base.util import slice_dict


class BaseCurvePredictor(Serializable):
  def __init__(self, **params):
    self._x = np.array([])
    self._y = np.array([])
    self._burn_in = params.get('burn_in', 10)
    self._min_input_size = params.get('min_input_size', 3)

    self._curve_io = DefaultIO(self, filename='curve-data.xjson', **slice_dict(params, 'io_'))
    self._curve_io.load()

  @property
  def curves_number(self):
    return self._x.shape[0]

  @property
  def curve_length(self):
    if self.curves_number == 0:
      return 0
    return self._x.shape[1]

  def add_curve(self, curve, value):
    assert len(curve.shape) == 1

    curve = curve.reshape(1, -1)
    value = value.reshape(1)

    if self.curves_number == 0:
      self._x = curve
      self._y = value
    else:
      self._x = np.concatenate([self._x, curve], axis=0)
      self._y = np.concatenate([self._y, value], axis=0)

    info('Adding curve (value=%.4f). Current data shape: ' % value[0], self._x.shape)
    self._curve_io.save()

  def predict(self, curve):
    raise NotImplementedError()

  def stop_condition(self):
    raise NotImplementedError()

  def result_metric(self):
    raise NotImplementedError()

  def import_from(self, data):
    self._import_from_keys(data, keys=('x', 'y'), default_value=[])

  def export_to(self):
    return self._export_keys_to(keys=('x', 'y'))


class LinearCurvePredictor(BaseCurvePredictor):
  def __init__(self, **params):
    super(LinearCurvePredictor, self).__init__(**params)
    self._value_limit = params.get('value_limit')
    self._model = {}

  def add_curve(self, curve, value):
    super(LinearCurvePredictor, self).add_curve(curve, value)
    self._model = {}

  def predict(self, curve):
    size = curve.shape[0]
    if self.curves_number < self._burn_in or \
       size < self._min_input_size or \
       size >= max(self.curves_number, self.curve_length):
      return None

    w, error = self._build_model(size)
    value_prediction = self._eval(curve[:size], w)
    value_prediction = max(value_prediction, max(curve))
    info('Prediction for the curve: %.4f (error=%.4f)' % (value_prediction, error))
    return value_prediction, error

  def stop_condition(self):
    def condition(curve):
      curve = np.array(curve)
      interval = self.predict(curve)
      if interval:
        expected, error = interval
        upper_bound = expected + error
        limit = self._value_limit or np.max(self._y)
        if upper_bound < limit:
          info('Max expected value for the curve is %.4f. Stop now (curve size=%d/%d)' %
               (upper_bound, curve.shape[0], self.curve_length))
          return True
      return False
    return condition

  def result_metric(self):
    def metric(curve):
      curve = np.array(curve)
      if curve.shape[0] < self.curve_length:
        expected, _ = self.predict(curve)
        info('Expected value for the curve is %.4f' % expected)
        return expected
      value = max(curve)
      self.add_curve(curve, value)
      return value
    return metric

  def _build_model(self, size):
    result = self._model.get(size)
    if result is None:
      w = self._compute_matrix(size)
      error = self._error(w, size)
      result = (w, error)
      self._model[size] = result
    return result

  def _compute_matrix(self, size):
    bias = np.ones(self.curves_number)
    x = np.column_stack([bias, self._x[:,:size]])
    y = self._y
    return np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

  def _error(self, w, size):
    x = self._x[:,:size]
    y = self._y
    predictions = self._eval(x, w)
    return np.max(np.abs(predictions - y))

  def _eval(self, x, w):
    if len(x.shape) > 1:
      bias = np.ones(x.shape[0])
      x = np.column_stack([bias, x])
    else:
      x = np.insert(x, 0, 1)
    return x.dot(w)
