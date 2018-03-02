#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils.layer_utils import count_params

from ...base import *
from ...model.base_solver import reducers


class MyCallback(Callback):
  def __init__(self):
    super(MyCallback, self).__init__()
    self._history = {}

  def history(self):
    return self._history

  def val_accuracy_history(self):
    return self._history['val_acc']

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    for k, v in logs.items():
      self._history.setdefault(k, []).append(v)

  def on_batch_begin(self, batch, logs=None):
    pass

  def on_batch_end(self, batch, logs=None):
    pass

  def on_train_begin(self, logs=None):
    pass

  def on_train_end(self, logs=None):
    pass


class KerasSolver(object):
  def __init__(self, model, data, hyper_params=None, model_io=None, reducer='max', **params):
    self._model = model

    self._train_set = data.train
    self._val_set = data.validation
    self._test_set = data.test
    self._hyper_params = hyper_params
    self._reducer = as_numeric_function(reducer, presets=reducers)

    self._model_io = model_io if model_io is not None else None

    self._epochs = params.get('epochs', 1)
    self._batch_size = params.get('batch_size', 16)
    self._eval_test = params.get('evaluate_test', False)


  def train(self):
    info('Start training. Model size: %dk' % (self._model_size() / 1000))
    info('Hyper params: %s' % smart_str(self._hyper_params))

    early_stopping = EarlyStopping()
    callback = MyCallback()
    self._model.fit(x=self._train_set.x,
                    y=self._train_set.y,
                    validation_data=(self._val_set.x, self._val_set.y),
                    batch_size=self._batch_size,
                    epochs=self._epochs,
                    verbose=1,
                    callbacks=[callback, early_stopping])

    return self._reducer(callback.val_accuracy_history())


  def terminate(self):
    K.clear_session()


  def _model_size(self):
    return count_params(self._model.trainable_weights)
