#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf

import hyperengine as hype
from common import get_wine_data

def dnn_model(params):
  x = tf.placeholder(shape=[None, 11], dtype=tf.float32, name='input')
  y = tf.placeholder(shape=[None], dtype=tf.float32, name='label')
  mode = tf.placeholder(tf.string, name='mode')
  training = tf.equal(mode, 'train')

  layer = tf.layers.batch_normalization(x, training=training) if params.batch_norm else x
  for hidden_size in params.hidden_layers:
    layer = tf.layers.dense(layer, units=hidden_size, activation=tf.nn.relu)
  layer = tf.layers.dropout(layer, params.dropout, training=training) if params.dropout else layer
  predictions = tf.layers.dense(layer, units=1)

  loss = tf.reduce_mean((predictions - y) ** 2, name='loss')
  optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
  optimizer.minimize(loss, name='minimize')

  # Hyper-tuner is generally designed for classification, so it tries to maximize the accuracy.
  # But we have a regression problem. A simple possible metric to emulate the accuracy would be:
  #
  #   tf.negative(loss, name='accuracy')
  #
  # But we'll use this one:
  tf.reduce_mean(tf.cast(tf.abs(predictions - y) < 0.5, tf.float32), name='accuracy')

x_train, y_train, x_test, y_test, x_val, y_val = get_wine_data(path='temp-wine/data')
data = hype.Data(train=hype.DataSet(x_train, y_train),
                 validation=hype.DataSet(x_val, y_val),
                 test=hype.DataSet(x_test, y_test))

def solver_generator(params):
  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 1000,
    'epochs': 20,
    'evaluate_test': True,
    'eval_flexible': True,
  }
  dnn_model(params)
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  batch_norm = hype.spec.random_bool(),
  hidden_layers = [hype.spec.choice(range(8, 33))],
  dropout = hype.spec.uniform(0.5, 1.0),
  learning_rate = 10**hype.spec.uniform(-1, -3),
)
strategy_params = {
  'io_load_dir': 'temp-wine/example-1-6',
  'io_save_dir': 'temp-wine/example-1-6',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
