#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np
from model.data_set import DataSet, merge_data_sets

import unittest


class DataSetTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    super(DataSetTest, cls).setUpClass()
    np.random.shuffle = lambda x : x

  def test_fit(self):
    arr = np.asanyarray([1, 2, 3, 4])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(2, [1, 2], 2, 2, 0, False)
    self.check_next_batch(2, [3, 4], 4, 0, 1, True)
    self.check_next_batch(2, [1, 2], 6, 2, 1, False)
    self.check_next_batch(2, [3, 4], 8, 0, 2, True)

  def test_fit2(self):
    arr = np.asanyarray([1, 2, 3, 4, 5, 6])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(2, [1, 2],  2, 2, 0, False)
    self.check_next_batch(2, [3, 4],  4, 4, 0, False)
    self.check_next_batch(2, [5, 6],  6, 0, 1, True)
    self.check_next_batch(2, [1, 2],  8, 2, 1, False)
    self.check_next_batch(2, [3, 4], 10, 4, 1, False)
    self.check_next_batch(2, [5, 6], 12, 0, 2, True)

  def test_does_not_fit(self):
    arr = np.asanyarray([1, 2, 3, 4, 5])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(2, [1, 2],  2, 2, 0, False)
    self.check_next_batch(2, [3, 4],  4, 4, 0, False)
    self.check_next_batch(2, [5,  ],  5, 0, 1, True)
    self.check_next_batch(2, [1, 2],  7, 2, 1, False)
    self.check_next_batch(2, [3, 4],  9, 4, 1, False)
    self.check_next_batch(2, [5,  ], 10, 0, 2, True)

  def test_too_small_batch(self):
    arr = np.asanyarray([1, 2, 3])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(4, [1, 2, 3], 3, 0, 1, True)
    self.check_next_batch(4, [1, 2, 3], 6, 0, 2, True)
    self.check_next_batch(4, [1, 2, 3], 9, 0, 3, True)

  def test_merge_data_sets(self):
    x_t = np.array([[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]]).reshape((-1, 2, 2, 1))
    y_t = np.array([0, 1])
    ds_t = DataSet(x_t, y_t)

    x_v = np.array([[[0, 0], [0, 0]]]).reshape((-1, 2, 2, 1))
    y_v = np.array([2])
    ds_v = DataSet(x_v, y_v)

    ds = merge_data_sets(ds_t, ds_v)
    self.assertEqual((3, 2, 2, 1), ds.x.shape)
    self.assertEqual((3, ), ds.y.shape)

    self.assert_equal_arrays(ds_t.x[0,:,:,0], ds.x[0,:,:,0])
    self.assert_equal_arrays(ds_t.x[1,:,:,0], ds.x[1,:,:,0])
    self.assert_equal_arrays(ds_v.x[0,:,:,0], ds.x[2,:,:,0])
    self.assert_equal_arrays([0, 1, 2], ds.y)

  def check_next_batch(self, batch_size, array, index, index_in_epoch, epochs_completed, just_completed):
    batch = self.ds.next_batch(batch_size)
    self.assertEquals(list(batch[0]), array)
    self.assertEquals(list(batch[1]), array)
    self.assertEquals(self.ds.index, index)
    self.assertEquals(self.ds.index_in_epoch, index_in_epoch)
    self.assertEquals(self.ds.epochs_completed, epochs_completed)
    self.assertEquals(self.ds.just_completed, just_completed)

  def assert_equal_arrays(self, array1, array2):
    self.assertTrue((array1 == array2).all())

if __name__ == '__main__':
  unittest.main()
