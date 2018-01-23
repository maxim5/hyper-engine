#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'maxim'

import tensorflow as tf
import hyperengine as hype

ACTIVATIONS = {name: getattr(tf.nn, name) for name in ['relu', 'relu6', 'elu', 'sigmoid', 'tanh', 'leaky_relu']}
DOWN_SAMPLES = {name: getattr(tf.layers, name) for name in ['max_pooling2d', 'average_pooling2d']}

def get_cifar10_data(validation_size=5000, one_hot=True):
  from tflearn.datasets import cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data('temp-cifar10-data', one_hot=one_hot)
  x_val = x_train[:validation_size]
  y_val = y_train[:validation_size]
  x_train = x_train[validation_size:]
  y_train = y_train[validation_size:]
  return hype.Data(train=hype.DataSet(x_train, y_train),
                   validation=hype.DataSet(x_val, y_val),
                   test=hype.DataSet(x_test, y_test))

def cnn_model(params):
  x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')
  y = tf.placeholder(tf.float32, [None, 10], name='label')
  mode = tf.placeholder(tf.string, name='mode')
  training = tf.equal(mode, 'train')

  def conv_layer(input, hp):
    conv = tf.layers.conv2d(input,
                            filters=hp.filter_num,
                            kernel_size=hp.filter_size,
                            padding='same', activation=ACTIVATIONS[hp.activation])
    bn = tf.layers.batch_normalization(conv, training=training) if hp.batch_norm else conv
    dropped = tf.layers.dropout(bn, rate=hp.dropout, training=training)
    pool = DOWN_SAMPLES[hp.down_sample](dropped, pool_size=[2, 2], strides=[2, 2])
    return pool

  def dense_layer(input, hp):
    flat = tf.reshape(input, [-1, input.shape[1] * input.shape[2] * input.shape[3]])
    dense = tf.layers.dense(inputs=flat, units=hp.size, activation=tf.nn.relu)
    dropped = tf.layers.dropout(dense, rate=hp.dropout, training=training)
    return dropped

  layer = conv_layer(x, params.conv[0])
  layer = conv_layer(layer, params.conv[1])
  layer = dense_layer(layer, params.dense)
  logits = tf.layers.dense(inputs=layer, units=10)

  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
  optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
  train_op = optimizer.minimize(loss_op, name='minimize')
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32), name='accuracy')

  return locals()  # to avoid GC

data = get_cifar10_data()

def solver_generator(params):
  solver_params = {
    'batch_size': 500,
    'eval_batch_size': 500,
    'epochs': 10,
    'evaluate_test': True,
    'eval_flexible': False,
    'save_dir': 'temp-cifar10-model-zoo/{date}-{random_id}',
    'save_accuracy_limit': 0.65,
  }
  cnn_model(params)
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  learning_rate = 10**hype.spec.uniform(-2, -3),
  conv = [
    # Layer 1
    hype.spec.new(
      filter_num = hype.spec.choice([32, 48, 64]),
      filter_size = [hype.spec.choice([3, 5])] * 2,
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = hype.spec.uniform(0.0, 0.2),
      down_sample = hype.spec.choice(DOWN_SAMPLES.keys()),
    ),
    # Layer 2
    hype.spec.new(
      filter_num = hype.spec.choice([96, 128]),
      filter_size = [hype.spec.choice([3, 5])] * 2,
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = hype.spec.uniform(0.0, 0.4),
      down_sample = hype.spec.choice(DOWN_SAMPLES.keys()),
    ),
  ],
  # Dense layer
  dense = hype.spec.new(
    size = hype.spec.choice([256, 512, 768, 1024]),
    dropout = hype.spec.uniform(0.0, 0.5),
  ),
)
strategy_params = {
  'io_load_dir': 'temp-cifar10-train/conv-1.0',
  'io_save_dir': 'temp-cifar10-train/conv-1.0',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
