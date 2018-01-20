#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import hyperengine as hype

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
y = tf.placeholder(tf.float32, [None, 10], name='label')

conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])

flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=dense, units=10)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op, name='minimize')
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32), name='accuracy')

tf_data_sets = input_data.read_data_sets('temp-mnist', one_hot=True)
convert = lambda data_set: hype.DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
data = hype.Data(train=convert(tf_data_sets.train),
                 validation=convert(tf_data_sets.validation),
                 test=convert(tf_data_sets.test))

solver_params = {
  'batch_size': 1000,
  'eval_batch_size': 2500,
  'epochs': 10,
  'evaluate_test': True,
  'eval_flexible': False,
}
solver = hype.TensorflowSolver(data=data, **solver_params)
solver.train()
