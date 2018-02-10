#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf

from ...base import *
from ...model import BaseSolver
from .tensorflow_model_io import TensorflowModelIO
from .tensorflow_runner import TensorflowRunner
from .tf_util import is_gpu_available


class TensorflowSolver(BaseSolver):
  def __init__(self, data, model=None, hyper_params=None, augmentation=None, model_io=None, reducer='max', **params):
    if isinstance(model, TensorflowRunner):
      runner = model
    else:
      runner = TensorflowRunner(model)

    self._session = None
    self._model_io = model_io if model_io is not None else TensorflowModelIO(**params)
    self._save_accuracy_limit = params.get('save_accuracy_limit', 0)

    params['eval_flexible'] = params.get('eval_flexible', True) and is_gpu_available()
    super(TensorflowSolver, self).__init__(runner, data, hyper_params, augmentation, reducer, **params)

  def create_session(self):
    self._session = tf.Session(graph=self._runner.graph())
    return self._session

  def init_session(self):
    self._runner.init(session=self._session)
    results = self._load(directory=self._model_io.load_dir, log_level=1)
    return results.get('validation_accuracy', 0)

  def terminate(self):
    self._runner.terminate()

  def on_best_accuracy(self, accuracy, eval_result):
    if accuracy >= self._save_accuracy_limit:
      self._model_io.save_results({'validation_accuracy': accuracy, 'model_size': self._runner.model_size()})
      self._model_io.save_hyper_params(self._hyper_params)
      self._model_io.save_session(self._session)
      self._model_io.save_data(eval_result.get('data'))

  def _evaluate_test(self):
    if not self._eval_test:
      return

    if self._test_set is None:
      warn('Test set is not provided. Skip test evaluation')
      return

    # Load the best session if available before test evaluation
    current_results = self._load(directory=self._model_io.save_dir, log_level=0)
    eval_ = super(TensorflowSolver, self)._evaluate_test()
    if not current_results:
      return eval_

    # Update the current results
    current_results['test_accuracy'] = eval_.get('accuracy', 0)
    self._model_io.save_results(current_results)
    return eval_

  def _load(self, directory, log_level):
    self._model_io.load_session(self._session, directory, log_level)
    results = self._model_io.load_results(directory, log_level)
    return results or {}
