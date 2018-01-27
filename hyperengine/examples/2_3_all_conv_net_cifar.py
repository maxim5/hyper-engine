#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf

from common import get_cifar10
import hyperengine as hype

ACTIVATIONS = {name: getattr(tf.nn, name) for name in ['relu', 'relu6', 'elu', 'sigmoid', 'tanh', 'leaky_relu']}
DOWN_SAMPLES = {name: getattr(tf.layers, name) for name in ['max_pooling2d', 'average_pooling2d']}

def cnn_model(params):
  x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')
  y = tf.placeholder(tf.float32, [None, 10], name='label')
  mode = tf.placeholder(tf.string, name='mode')
  training = tf.equal(mode, 'train')

  def conv_layer(input, hp):
    layer = input
    for filter in hp.filters:
      layer = tf.layers.conv2d(layer,
                               filters=filter[-1],
                               kernel_size=filter[:-1],
                               padding=hp.get('padding', 'same'),
                               activation=ACTIVATIONS.get(hp.activation, None))
    layer = tf.layers.batch_normalization(layer, training=training) if hp.batch_norm else layer
    layer = tf.layers.dropout(layer, rate=hp.dropout, training=training) if hp.dropout else layer
    layer = DOWN_SAMPLES[hp.down_sample](layer, pool_size=[2, 2], strides=[2, 2]) if hp.down_sample else layer
    return layer

  layer = x
  for conv_hp in params.conv:
    layer = conv_layer(layer, hp=conv_hp)
  logits = tf.squeeze(layer, axis=[1, 2])

  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
  optimizer = tf.train.AdamOptimizer(learning_rate=params.optimizer.learning_rate,
                                     beta1=params.optimizer.beta1,
                                     beta2=params.optimizer.beta2,
                                     epsilon=params.optimizer.epsilon)
  train_op = optimizer.minimize(loss_op, name='minimize')
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32), name='accuracy')

  return locals()  # to avoid GC

(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_cifar10(path='temp-cifar10/data',
                                                                   one_hot=True,
                                                                   validation_size=5000)
data = hype.Data(train=hype.DataSet(x_train, y_train),
                 validation=hype.DataSet(x_val, y_val),
                 test=hype.DataSet(x_test, y_test))


def solver_generator(params):
  solver_params = {
    'batch_size': 500,
    'eval_batch_size': 500,
    'epochs': 10,
    'evaluate_test': True,
    'eval_flexible': False,
    'save_dir': 'temp-cifar10/model-zoo/example-2-3-{date}-{random_id}',
    'save_accuracy_limit': 0.70,
  }
  cnn_model(params)
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  optimizer = hype.spec.new(
    learning_rate = 10**hype.spec.uniform(-2, -4),
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
  ),
  conv = [
    # Layer 1: accepts 32x32
    hype.spec.new(
      filters = [[3, 3, hype.spec.choice([48, 64])]],
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = None,
      down_sample = hype.spec.choice(DOWN_SAMPLES.keys()),
    ),
    # Layer 2: accepts 16x16
    hype.spec.new(
      filters = [[3, 3, hype.spec.choice([64, 96, 128])]],
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = None,
      down_sample = hype.spec.choice(DOWN_SAMPLES.keys()),
    ),
    # Layer 3: accepts 8x8
    hype.spec.new(
      filters = [[1, 1, 64],
                 [3, 3, hype.spec.choice([128, 192, 256])]],
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = None,
      down_sample = hype.spec.choice(DOWN_SAMPLES.keys()),
    ),
    # Layer 4: accepts 4x4
    hype.spec.new(
      filters = [[1, 1, 96],
                 [2, 2, hype.spec.choice([256, 384, 512])]],
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = None,
      down_sample = hype.spec.choice(DOWN_SAMPLES.keys()),
    ),
    # Layer 5: accepts 2x2
    hype.spec.new(
      filters = [[1, 1, 128]],
    ),
    # Layer 6: accepts 2x2
    hype.spec.new(
      filters = [[2, 2, hype.spec.choice([512, 768])]],
      padding = 'valid',
      activation = hype.spec.choice(ACTIVATIONS.keys()),
      batch_norm = False,
      dropout = None,
    ),
    # Layer 7: accepts 1x1
    hype.spec.new(
      filters=[[1, 1, 10]],
    ),
  ],
)
strategy_params = {
  'io_load_dir': 'temp-cifar10/example-2-3',
  'io_save_dir': 'temp-cifar10/example-2-3',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
