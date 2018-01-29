#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import inspect

from ..base import *
from .data_set import IterableDataProvider

metrics = {
  'max': lambda curve: max(curve) if curve else 0,
  'avg': lambda curve: sum(curve) / len(curve) if curve else 0,
}

class BaseSolver(object):
  """
  Implements a training algorithm.
  """

  def __init__(self, runner, data, hyper_params, augmentation=None, result_metric='max', **params):
    data.reset_counters()
    self._train_set = data.train
    self._val_set = data.validation
    self._test_set = data.test
    self._augmentation = augmentation
    self._runner = runner
    self._hyper_params = hyper_params
    self._result_metric = as_numeric_function(result_metric, presets=metrics)
    self._max_val_accuracy = 0
    self._val_accuracy_curve = []

    self._epochs = params.get('epochs', 1)
    self._dynamic_epochs = params.get('dynamic_epochs')
    self._stop_condition = params.get('stop_condition')
    self._batch_size = params.get('batch_size', 16)
    self._eval_batch_size = params.get('eval_batch_size', self._val_set.size if self._val_set else 0)
    self._eval_flexible = params.get('eval_flexible', True)
    self._eval_train_every = params.get('eval_train_every', 10) if not self._eval_flexible else 1e1000
    self._eval_validation_every = params.get('eval_validation_every', 100) if not self._eval_flexible else 1e1000
    self._eval_test = params.get('evaluate_test', False)

  def train(self):
    self._runner.build_model()
    info('Start training. Model size: %dk' % (self._runner.model_size() / 1000))
    info('Hyper params: %s' % smart_str(self._hyper_params))

    with self.create_session():
      self._max_val_accuracy = self.init_session()
      while self._train_set.epochs_completed < self._epochs:
        batch_x, batch_y = self._train_set.next_batch(self._batch_size)
        batch_x = self.augment(batch_x)
        self._runner.run_batch(batch_x, batch_y)
        self._evaluate_train(batch_x, batch_y)

        eval_result = self._evaluate_validation()
        val_accuracy = eval_result.get('accuracy') if eval_result is not None else None
        if val_accuracy is not None:
          self._val_accuracy_curve.append(val_accuracy)
          self._update_epochs_dynamically()

          if val_accuracy > self._max_val_accuracy:
            self._max_val_accuracy = val_accuracy
            self.on_best_accuracy(val_accuracy, eval_result)

          if self._stop_condition and self._stop_condition(self._val_accuracy_curve):
            info('Solver stopped due to the stop condition')
            break

      self._evaluate_test()
    return self._result_metric(self._val_accuracy_curve)

  def create_session(self):
    return None

  def init_session(self):
    return 0

  def augment(self, x):
    augmented = call(self._augmentation, x)
    return augmented if augmented is not None else x

  def terminate(self):
    pass

  def on_best_accuracy(self, accuracy, eval_result):
    pass

  def _update_epochs_dynamically(self):
    if self._dynamic_epochs is None:
      return

    curve = self._val_accuracy_curve
    max_acc = max(self._val_accuracy_curve)

    new_epochs = self._epochs
    args_spec = inspect.getargspec(self._dynamic_epochs)
    if args_spec.varargs is not None or args_spec.keywords is not None:
      new_epochs = self._dynamic_epochs(curve, max_acc)
    else:
      args = args_spec.args
      if len(args) == 1:
        arg = self._val_accuracy_curve if args[0] in ['curve', 'history'] else max_acc
        new_epochs = self._dynamic_epochs(arg)
      elif args == 2:
        new_epochs = self._dynamic_epochs(curve)
      else:
        warn('Invalid "dynamic_epochs" parameter: '
             'expected a function with either one or two arguments, but got %s' % args_spec)

    if self._epochs != new_epochs:
      self._epochs = new_epochs or self._epochs
      debug('Update epochs=%d' % new_epochs)

  def _evaluate_train(self, batch_x, batch_y):
    eval_this_step = self._train_set.step % self._eval_train_every == 0
    if eval_this_step and is_info_logged():
      eval_ = self._runner.evaluate(batch_x, batch_y)
      self._log_iteration('train_accuracy', eval_.get('loss'), eval_.get('accuracy'), False)

  def _evaluate_validation(self):
    eval_this_step = self._train_set.step % self._eval_validation_every == 0
    epoch_just_completed = (self._train_set.just_completed and self._eval_flexible)

    if eval_this_step or epoch_just_completed:
      if self._val_set is None:
        warn('Validation set is not provided. Skip val evaluation')
        return

      eval_ = self._evaluate(self._val_set)
      self._log_iteration('validation_accuracy', eval_.get('loss'), eval_.get('accuracy'), True)
      return eval_

  def _evaluate_test(self):
    if not self._eval_test:
      return

    if self._test_set is None:
      warn('Test set is not provided. Skip test evaluation')
      return

    eval_ = self._evaluate(self._test_set)
    test_accuracy = eval_.get('accuracy')
    if test_accuracy is None:
      warn('Test accuracy evaluation is not available')
      return

    info('Final test_accuracy=%.4f' % test_accuracy)
    return eval_

  def _log_iteration(self, name, loss, accuracy, mark_best):
    message = 'Epoch %2d, iteration %7d' % (self._train_set.epochs_completed, self._train_set.index)
    if accuracy is not None:
      marker = ' *' if mark_best and (accuracy > self._max_val_accuracy) else ''
      if loss is None:
        info('%s: %s=%.4f%s' % (message, name, accuracy, marker))
      else:
        info('%s: loss=%.6f, %s=%.4f%s' % (message, loss, name, accuracy, marker))
    else:
      if loss is not None:
        info('%s: loss=%.6f' % (message, loss))
      else:
        info('%s: -- no loss or accuracy defined --' % message)

  def _evaluate(self, data_set):
    assert isinstance(data_set, IterableDataProvider), 'Currently can evaluate only IterableDataProviders'
    size = data_set.size
    data_set.reset_counters()

    result = {'accuracy': 0, 'loss': 0, 'misclassified_x': [], 'misclassified_y': []}
    while data_set.epochs_completed < 1:
      batch_x, batch_y = data_set.next_batch(self._eval_batch_size)
      eval_ = self._runner.evaluate(batch_x=batch_x, batch_y=batch_y)
      result['accuracy'] += eval_ .get('accuracy', 0) * len(batch_x)
      result['loss'] += eval_ .get('loss', 0) * len(batch_x)
      result['misclassified_x'].append(eval_ .get('misclassified_x'))
      result['misclassified_y'].append(eval_ .get('misclassified_y'))
    result['accuracy'] /= size
    result['loss'] /= size
    result['misclassified_x'] = safe_concat(result['misclassified_x'])
    result['misclassified_y'] = safe_concat(result['misclassified_y'])

    return result
