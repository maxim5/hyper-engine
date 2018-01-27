#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

from collections import Counter, deque

import numpy as np
import tensorflow as tf

from common import get_text8
import hyperengine as hype


class SentenceDataProvider(hype.IterableDataProvider):
  def __init__(self, **params):
    super(SentenceDataProvider, self).__init__()

    self._vocab_size = params.get('vocab_size', 50000)
    self._num_skips = params.get('num_skips', 2)
    self._skip_window = params.get('skip_window', 1)

    self._step = 0
    self._index = 0
    self._epochs_completed = 0
    self._just_completed = False

    self._vocabulary = None
    self._dictionary = None
    self._data = None

  @property
  def size(self):
    return len(self._data)

  @property
  def vocabulary(self):
    return self._vocabulary

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def step(self):
    return self._step

  @property
  def index(self):
    return self._index

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def just_completed(self):
    return self._just_completed

  def reset_counters(self):
    self._step = 0
    self._index = 0
    self._epochs_completed = 0
    self._just_completed = False

  def build(self):
    hype.util.debug('Building the data provider')
    words = get_text8('temp-text8/data')
    self._vocabulary = [('UNK', None)] + Counter(words).most_common(self._vocab_size - 1)
    self._vocabulary = np.array([word for word, _ in self._vocabulary])
    self._dictionary = {word: code for code, word in enumerate(self._vocabulary)}
    self._data = np.array([self._dictionary.get(word, 0) for word in words])

    if hype.util.is_debug_logged():
      hype.util.debug('Total words in text: %dM' % (len(words) / 1000000))
      hype.util.debug('Text example: %s...' % ' '.join(words[:10]))
      hype.util.debug('Encoded text: %s...' % self._data[:10].tolist())

  def next_batch(self, batch_size):
    self._step += 1
    return self._generate_batch(batch_size, self._num_skips, self._skip_window)

  def _generate_batch(self, batch_size, num_skips, skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size, ), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)
    for _ in range(span):
      buffer.append(self._data[self._index])
      self._inc_index()
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = np.random.randint(0, span)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(self._data[self._index])
      self._inc_index()
    return batch, labels

  def _inc_index(self):
    next = self._index + 1
    if next >= len(self._data):
      self._index = 0
      self._epochs_completed += 1
      self._just_completed = True
    else:
      self._index = next
      self._just_completed = False


def word2vec_model(params):
  # Input data.
  inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='input')
  labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='label')

  # Look up embeddings for inputs.
  embeddings = tf.Variable(
    tf.random_uniform([params.vocab_size, params.embedding_size], -1.0, 1.0)
  )
  embed = tf.nn.embedding_lookup(embeddings, inputs)

  # Construct the variables for the NCE loss
  nce_weights = tf.Variable(
    tf.truncated_normal(shape=[params.vocab_size, params.embedding_size],
                        stddev=1.0 / np.sqrt(params.embedding_size))
  )
  nce_biases = tf.Variable(tf.zeros([params.vocab_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases,
                   labels=labels,
                   inputs=embed,
                   num_sampled=params.negative_samples,
                   num_classes=params.vocab_size),
    name='loss'
  )
  optimizer = tf.train.AdamOptimizer(params.learning_rate)
  optimizer.minimize(loss, name='minimize')


provider = SentenceDataProvider()
provider.build()
data = hype.Data(train=provider, validation=None, test=None)

word2vec_model(params=hype.spec.new(
  vocab_size = provider.vocab_size,
  embedding_size = 128,
  negative_samples = 64,
  learning_rate = 0.01,
))

solver_params = {
  'batch_size': 1024,
  'epochs': 5,
  'eval_flexible': False,
}
solver = hype.TensorflowSolver(data=data, **solver_params)
solver.train()
