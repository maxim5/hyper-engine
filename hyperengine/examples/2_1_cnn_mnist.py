#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import hyperengine as hype

ACTIVATIONS = {name: getattr(tf.nn, name) for name in ['relu', 'relu6', 'elu', 'sigmoid', 'tanh', 'leaky_relu']}

def cnn_model(params):
  x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
  y = tf.placeholder(tf.float32, [None, 10], name='label')
  mode = tf.placeholder(tf.string, name='mode')
  training = tf.equal(mode, 'train')

  def conv_layer(input, hp):
    conv = tf.layers.conv2d(input,
                            filters=hp.filter_num,
                            kernel_size=hp.filter_size,
                            padding='same',
                            activation=ACTIVATIONS[hp.activation])
    bn = tf.layers.batch_normalization(conv, training=training) if hp.batch_norm else conv
    dropped = tf.layers.dropout(bn, rate=hp.dropout, training=training) if hp.dropout else bn
    pool = tf.layers.max_pooling2d(dropped, pool_size=[2, 2], strides=[2, 2])
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

tf_data_sets = input_data.read_data_sets('temp-mnist/data', one_hot=True)
convert = lambda data_set: hype.DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
data = hype.Data(train=convert(tf_data_sets.train),
                 validation=convert(tf_data_sets.validation),
                 test=convert(tf_data_sets.test))

def solver_generator(params):
  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 2500,
    'epochs': 10,
    'evaluate_test': True,
    'eval_flexible': False,
    'save_dir': 'temp-mnist/model-zoo/example-2-1-{date}-{random_id}',
    'save_accuracy_limit': 0.9930,
  }
  cnn_model(params)
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  learning_rate = 10**hype.spec.uniform(-2, -3),
  conv = [
    # Layer 1
    hype.spec.new(
      filter_num = hype.spec.choice([20, 32, 48]),
      filter_size = [hype.spec.choice([3, 5])] * 2,
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = hype.spec.random_bool(),
      dropout = hype.spec.uniform(0.0, 0.3),
    ),
    # Layer 2
    hype.spec.new(
      filter_num = hype.spec.choice([64, 96, 128]),
      filter_size = [hype.spec.choice([3, 5])] * 2,
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = hype.spec.random_bool(),
      dropout = hype.spec.uniform(0.0, 0.5),
    ),
  ],
  # Dense layer
  dense = hype.spec.new(
    size = hype.spec.choice([256, 512, 768, 1024]),
    dropout = hype.spec.uniform(0.0, 0.5),
  ),
)
strategy_params = {
  'io_load_dir': 'temp-mnist/example-2-1',
  'io_save_dir': 'temp-mnist/example-2-1',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
