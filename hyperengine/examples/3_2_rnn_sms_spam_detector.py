#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import preprocessing

from common import get_sms_spam_data
import hyperengine as hype


inputs, labels = get_sms_spam_data('temp-sms-spam/data')
hype.util.info('Data examples:')
for text, target in zip(inputs[:15], labels[:15]):
  hype.util.info('%4s: %s' % (target, text[:100]))
hype.util.info()


def preprocess(params):
  # Data processing: encode text inputs into numeric vectors
  processor = preprocessing.VocabularyProcessor(max_document_length=params.max_sequence_length,
                                                min_frequency=params.min_frequency)
  encoded_inputs = list(processor.fit_transform(inputs))
  vocab_size = len(processor.vocabulary_)

  # Set this to see verbose output:
  #   hype.util.set_verbose()

  if hype.util.is_debug_logged():
    hype.util.debug('Encoded text examples:')
    for i in range(3):
      hype.util.debug('  %s ->' % inputs[i])
      hype.util.debug('  %s\n' % encoded_inputs[i].tolist())

  encoded_inputs = np.array(encoded_inputs)
  encoded_labels = np.array([int(label == 'ham') for label in labels])

  # Shuffle the data
  shuffled_ix = np.random.permutation(np.arange(len(encoded_labels)))
  x_shuffled = encoded_inputs[shuffled_ix]
  y_shuffled = encoded_labels[shuffled_ix]

  # Split into train/validation/test sets
  idx1 = int(len(y_shuffled) * 0.75)
  idx2 = int(len(y_shuffled) * 0.85)
  x_train, x_val, x_test = x_shuffled[:idx1], x_shuffled[idx1:idx2], x_shuffled[idx2:]
  y_train, y_val, y_test = y_shuffled[:idx1], y_shuffled[idx1:idx2], y_shuffled[idx2:]

  if hype.util.is_debug_logged():
    hype.util.debug('Vocabulary size: %d' % vocab_size)
    hype.util.debug('Train/validation/test split: train=%d, val=%d, test=%d' %
                    (len(y_train), len(y_val), len(y_test)))

  train = hype.DataSet(x_train, y_train)
  validation = hype.DataSet(x_val, y_val)
  test = hype.DataSet(x_test, y_test)
  data = hype.Data(train, validation, test)
  
  return data, vocab_size


def rnn_model(params):
  data, vocab_size = preprocess(params)

  x = tf.placeholder(shape=[None, params.max_sequence_length], dtype=tf.int32, name='input')
  y = tf.placeholder(shape=[None], dtype=tf.int32, name='label')
  mode = tf.placeholder(tf.string, name='mode')
  training = tf.equal(mode, 'train')

  # Create embedding
  embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, params.embedding_size], -1.0, 1.0))
  embedding_output = tf.nn.embedding_lookup(embedding_matrix, x)

  if params.rnn_cell == 'basic_rnn':
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=params.rnn_hidden_size)
  elif params.rnn_cell == 'lstm':
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params.rnn_hidden_size)
  elif params.rnn_cell == 'gru':
    cell = tf.nn.rnn_cell.GRUCell(num_units=params.rnn_hidden_size)
  else:
    raise ValueError('Unexpected hyper-parameter: %s' % params.rnn_cell)
  output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
  output = tf.cond(training, lambda: tf.nn.dropout(output, keep_prob=params.dropout_keep_prob), lambda: output)

  # Get output of RNN sequence
  # output = (?, max_sequence_length, rnn_hidden_size)
  # output_last = (?, rnn_hidden_size)
  output = tf.transpose(output, [1, 0, 2])
  output_last = tf.gather(output, int(output.get_shape()[0]) - 1)

  # Final logits for binary classification:
  # logits = (?, 2)
  weight = tf.Variable(tf.truncated_normal([params.rnn_hidden_size, 2], stddev=0.1))
  bias = tf.Variable(tf.constant(0.1, shape=[2]))
  logits = tf.matmul(output_last, weight) + bias

  # Loss function
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
  loss = tf.reduce_mean(cross_entropy, name='loss')
  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64)), tf.float32), name='accuracy')
  optimizer = tf.train.RMSPropOptimizer(params.learning_rate)
  optimizer.minimize(loss, name='minimize')
  
  return data


def solver_generator(params):
  solver_params = {
    'batch_size': 200,
    'eval_batch_size': 200,
    'epochs': 20,
    'evaluate_test': True,
    'eval_flexible': False,
    'save_dir': 'temp-sms-spam/model-zoo/example-3-2-{date}-{random_id}',
    'save_accuracy_limit': 0.97,
  }
  data = rnn_model(params)
  solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
  return solver

hyper_params_spec = hype.spec.new(
  max_sequence_length = hype.spec.choice(range(20, 50)),
  min_frequency = hype.spec.choice([1, 3, 5, 10]),
  embedding_size = hype.spec.choice([32, 64, 128]),
  rnn_cell = hype.spec.choice(['basic_rnn', 'lstm', 'gru']),
  rnn_hidden_size = hype.spec.choice([16, 32, 64]),
  dropout_keep_prob = hype.spec.uniform(0.5, 1.0),
  learning_rate = 10**hype.spec.uniform(-4, -3),
)

strategy_params = {
  'io_load_dir': 'temp-sms-spam/example-3-2',
  'io_save_dir': 'temp-sms-spam/example-3-2',
}
tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
