#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf

import hyperengine as hype


def rnn_model(params):
  x = tf.placeholder(tf.float32, [None, 28, 28], name='input')
  y = tf.placeholder(tf.int32, [None], name='label')

  lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=layer) for layer in params.lstm.layers]
  multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
  outputs, states = tf.nn.dynamic_rnn(multi_cell, x, dtype=tf.float32)

  # Here, `states` holds the final states of 3 layers, `states[-1]` is the state of the last layer.
  # Hence the name: `h` state of the top layer (short-term state)
  top_layer_h_state = states[-1][1]
  logits = tf.layers.dense(top_layer_h_state, 10)
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
  loss = tf.reduce_mean(xentropy, name='loss')
  optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
  optimizer.minimize(loss, name='minimize')
  correct = tf.nn.in_top_k(logits, y, 1)
  tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

from tensorflow.examples.tutorials.mnist import input_data
tf_data_sets = input_data.read_data_sets('temp-mnist/data', one_hot=False)
convert = lambda data_set: hype.DataSet(data_set.images.reshape((-1, 28, 28)), data_set.labels)
data = hype.Data(train=convert(tf_data_sets.train),
                 validation=convert(tf_data_sets.validation),
                 test=convert(tf_data_sets.test))

def solver_generator(params):
  rnn_model(params)

  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 2500,
    'epochs': 10,
    'evaluate_test': True,
    'eval_flexible': False,
    'save_dir': 'temp-mnist/model-zoo/example-3-1-{date}-{random_id}',
    'save_accuracy_limit': 0.99,
  }
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  learning_rate = 10**hype.spec.uniform(-2, -3),
  lstm = hype.spec.new(
    layers = [hype.spec.choice([128, 160, 256]),
              hype.spec.choice([128, 160, 256]),
              hype.spec.choice([128, 160, 256])]
  )
)
strategy_params = {
  'io_load_dir': 'temp-mnist/example-3-1',
  'io_save_dir': 'temp-mnist/example-3-1',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
