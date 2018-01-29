#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np


class DataProvider(object):
  pass


class IterableDataProvider(DataProvider):
  def __init__(self):
    super(IterableDataProvider, self).__init__()
    self._size = 0
    self._step = 0
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._just_completed = False

  @property
  def size(self):
    """
    Data size (number of rows)
    """
    return self._size

  @property
  def step(self):
    """
    The number of batches processed
    """
    return self._step

  @property
  def index(self):
    """
    Total index of input rows (over all epochs)
    """
    return self._epochs_completed * self._size + self._index_in_epoch

  @property
  def index_in_epoch(self):
    """
    The index of input rows in a current epoch
    """
    return self._index_in_epoch

  @property
  def epochs_completed(self):
    """
    A number of completed epochs
    """
    return self._epochs_completed

  @property
  def just_completed(self):
    """
    Whether the previous epoch was just completed
    """
    return self._just_completed

  def reset_counters(self):
    """
    Resets all counters.
    """
    self._step = 0
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._just_completed = False

  def next_batch(self, batch_size):
    """
    Returns the next `batch_size` examples from this data set.
    """
    raise NotImplementedError

  def _inc_index(self):
    index = self._index_in_epoch + 1
    if index >= self._size:
      self._index_in_epoch = 0
      self._epochs_completed += 1
      self._just_completed = True
    else:
      self._index_in_epoch = index
      self._just_completed = False


class DataSet(IterableDataProvider):
  """
  A labeled data set. Both inputs and labels are stored as numpy arrays in memory.
  """

  def __init__(self, x, y):
    super(DataSet, self).__init__()

    x = np.array(x)
    y = np.array(y)
    assert x.shape[0] == y.shape[0]

    self._size = x.shape[0]
    self._x = x
    self._y = y

  @property
  def x(self):
    return self._x

  @property
  def y(self):
    return self._y

  def next_batch(self, batch_size):
    if self._just_completed:
      permutation = np.arange(self._size)
      np.random.shuffle(permutation)
      self._x = self._x[permutation]
      self._y = self._y[permutation]

    self._step += 1
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    end = min(self._index_in_epoch, self._size)
    if self._index_in_epoch >= self._size:
      self._index_in_epoch = 0
    self._just_completed = end == self._size
    self._epochs_completed += int(self._just_completed)
    return self._x[start:end], self._y[start:end]


def merge_data_sets(ds1, ds2):
  x = np.concatenate([ds1.x, ds2.x], axis=0)
  y = np.concatenate([ds1.y, ds2.y], axis=0)
  return DataSet(x, y)


class Data(object):
  """
  Holds a standard data division: training set, validation set and test set.
  """

  def __init__(self, train, validation, test):
    assert train is not None, 'Training set must be not None'
    assert train is not validation, 'Validation set can not coincide with the training set'
    assert train is not test, 'Test set can not coincide with the training set'

    self.train = train
    self.validation = validation
    self.test = test

  def reset_counters(self):
    self.train.reset_counters()
    if self.validation:
      self.validation.reset_counters()
    if self.test:
      self.test.reset_counters()
