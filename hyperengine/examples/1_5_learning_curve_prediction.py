#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import hyperengine as hype

def cnn_model(params):
  x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
  y = tf.placeholder(tf.float32, [None, 10], name='label')
  mode = tf.placeholder(tf.string, name='mode')

  conv1 = tf.layers.conv2d(x, filters=params.conv_filters_1, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

  conv2 = tf.layers.conv2d(pool1, filters=params.conv_filters_2, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])

  flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
  dense = tf.layers.dense(inputs=flat, units=params.dense_size, activation=tf.nn.relu)
  dense = tf.layers.dropout(dense, rate=params.dropout_rate, training=tf.equal(mode, 'train'))
  logits = tf.layers.dense(inputs=dense, units=10)

  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
  optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
  train_op = optimizer.minimize(loss_op, name='minimize')
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32), name='accuracy')

  return locals()  # to avoid GC

tf_data_sets = input_data.read_data_sets('temp-mnist/data', one_hot=True)
convert = lambda data_set: hype.DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
data = hype.Data(train=convert(tf_data_sets.train),
                 validation=convert(tf_data_sets.validation),
                 test=convert(tf_data_sets.test))

curve_params = {
  'burn_in': 20,
  'min_input_size': 4,
  'value_limit': 0.82,
  'io_load_dir': 'temp-mnist/example-1-5',
  'io_save_dir': 'temp-mnist/example-1-5',
}
curve_predictor = hype.LinearCurvePredictor(**curve_params)

def solver_generator(params):
  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 2500,
    'epochs': 10,
    'stop_condition': curve_predictor.stop_condition(),
    'reducer': curve_predictor.result_metric(),
    'evaluate_test': True,
    'eval_flexible': False,
    'save_dir': 'temp-mnist/model-zoo/example-1-5-{date}-{random_id}',
    'save_accuracy_limit': 0.9930,
  }
  cnn_model(params)
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  learning_rate = 10**hype.spec.uniform(-2, -3),
  conv_filters_1 = hype.spec.choice([20, 32, 48]),
  conv_filters_2 = hype.spec.choice([64, 96, 128]),
  dense_size = hype.spec.choice([256, 512, 768, 1024]),
  dropout_rate = hype.spec.uniform(0.5, 0.9),
)
strategy_params = {
  'io_load_dir': 'temp-mnist/example-1-5',
  'io_save_dir': 'temp-mnist/example-1-5',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
