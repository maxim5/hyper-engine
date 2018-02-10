#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import unittest

import numpy as np

from hyperengine.base import *
from hyperengine import LinearCurvePredictor

SYNTHETIC_TRAIN = [
  ([40, 50, 55, 60, 63, 65, 68, 67, 69, 70, 72], 72),
  ([30, 34, 40, 42, 46, 46, 50, 52, 53, 55, 54], 55),
  ([32, 40, 45, 48, 54, 56, 57, 58, 60, 61, 61], 61),
  ([37, 45, 51, 54, 57, 60, 59, 63, 64, 67, 68], 68),
  ([44, 48, 53, 59, 63, 63, 64, 64, 68, 69, 70], 70),
  ([42, 49, 53, 57, 60, 63, 62, 65, 66, 68, 68], 68),
  ([41, 47, 55, 58, 61, 62, 65, 67, 69, 71, 71], 71),
  ([36, 43, 50, 52, 57, 61, 61, 62, 64, 66, 69], 69),
  ([20, 25, 30, 34, 38, 42, 46, 50, 53, 55, 54], 55),
  ([25, 27, 35, 40, 43, 47, 51, 52, 55, 56, 57], 57),
  ([29, 34, 40, 45, 50, 53, 56, 57, 57, 60, 59], 60),
  ([10, 16, 25, 30, 32, 35, 36, 35, 37, 37, 37], 37),
  ([41, 51, 52, 56, 61, 62, 65, 66, 68, 67, 69], 69),
  ([45, 48, 54, 57, 60, 62, 62, 65, 65, 66, 66], 66),
  ([22, 30, 33, 37, 39, 40, 45, 47, 48, 48, 51], 51),
  ([42, 51, 55, 56, 58, 61, 62, 64, 65, 66, 68], 68),
  ([34, 43, 45, 50, 52, 55, 61, 61, 60, 66, 64], 66),
  ([20, 24, 25, 30, 31, 33, 36, 37, 40, 42, 43], 43),
  ([44, 51, 55, 58, 60, 63, 65, 64, 66, 67, 67], 67),
  ([41, 49, 55, 58, 61, 61, 66, 67, 68, 68, 71], 71),
  ([34, 45, 48, 53, 55, 57, 61, 62, 63, 63, 64], 64),
]
SYNTHETIC_TEST = [
  ([43, 51, 55, 57, 62, 64, 67, 66, 68, 68, 71], 71),
  ([42, 50, 53, 59, 61, 63, 64, 66, 67, 68, 70], 70),
  ([39, 46, 51, 54, 59, 61, 62, 63, 65, 65, 66], 66),
  ([41, 51, 52, 56, 61, 62, 64, 66, 68, 68, 69], 69),
  ([44, 48, 54, 57, 60, 62, 65, 65, 66, 66, 68], 68),
  ([40, 51, 54, 58, 62, 66, 68, 69, 71, 70, 72], 72),
]


class LinearCurvePredictorTest(unittest.TestCase):
  def test_synthetic(self):
    self.set = SYNTHETIC_TRAIN, SYNTHETIC_TEST
    self.check_prediction(size=3, max_diff=2.5, avg_diff=1.2)
    self.check_prediction(size=4, max_diff=2.0, avg_diff=1.2)
    self.check_prediction(size=5, max_diff=2.5, avg_diff=1.2)
    self.check_prediction(size=6, max_diff=2.5, avg_diff=1.2)
    self.check_prediction(size=7, max_diff=3.0, avg_diff=1.8)
    self.check_prediction(size=8, max_diff=1.5, avg_diff=1.0)
    self.check_prediction(size=9, max_diff=2.5, avg_diff=1.2)

  def test_cifar10_1(self):
    self.set = self.read_curve_data(train_size=20, test_size=10)
    self.check_prediction(size=3,  max_diff=0.07, avg_diff=0.03, check_between=False)
    self.check_prediction(size=4,  max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=5,  max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=6,  max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=7,  max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=8,  max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=9,  max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=10, max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=11, max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=12, max_diff=0.05, avg_diff=0.03, check_between=False)
    self.check_prediction(size=13, max_diff=0.07, avg_diff=0.04, check_between=False)
    self.check_prediction(size=14, max_diff=0.07, avg_diff=0.04, check_between=False)
    self.check_prediction(size=15, max_diff=0.06, avg_diff=0.04, check_between=False)
    self.check_prediction(size=16, max_diff=0.08, avg_diff=0.04, check_between=False)
    self.check_prediction(size=17, max_diff=0.06, avg_diff=0.04, check_between=False)
    self.check_prediction(size=18, max_diff=0.06, avg_diff=0.04, check_between=False)
    self.check_prediction(size=19, max_diff=0.20, avg_diff=0.08, check_between=False)
    self.check_prediction(size=20, max_diff=0.07, avg_diff=0.03, check_between=False)

  def test_cifar10_2(self):
    self.set = self.read_curve_data(train_size=29, test_size=10)
    self.check_prediction(size=3,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=4,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=5,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=6,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=7,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=8,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=9,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=10, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=11, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=12, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=13, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=14, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=15, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=16, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=17, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=18, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=19, max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=20, max_diff=0.04, avg_diff=0.02, check_between=False)

  def test_cifar10_3(self):
    self.set = self.read_curve_data(train_size=50, test_size=10)
    self.check_prediction(size=3,  max_diff=0.03, avg_diff=0.02, check_between=False)
    self.check_prediction(size=4,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=5,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=6,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=7,  max_diff=0.04, avg_diff=0.02, check_between=False)
    self.check_prediction(size=8,  max_diff=0.03, avg_diff=0.02, check_between=False)
    self.check_prediction(size=9,  max_diff=0.03, avg_diff=0.01, check_between=False)
    self.check_prediction(size=10, max_diff=0.03, avg_diff=0.01, check_between=False)
    self.check_prediction(size=11, max_diff=0.03, avg_diff=0.02, check_between=False)
    self.check_prediction(size=12, max_diff=0.03, avg_diff=0.02, check_between=False)
    self.check_prediction(size=13, max_diff=0.03, avg_diff=0.02, check_between=False)
    self.check_prediction(size=14, max_diff=0.02, avg_diff=0.01, check_between=False)
    self.check_prediction(size=15, max_diff=0.02, avg_diff=0.01, check_between=False)
    self.check_prediction(size=16, max_diff=0.02, avg_diff=0.01, check_between=False)
    self.check_prediction(size=17, max_diff=0.02, avg_diff=0.01, check_between=False)
    self.check_prediction(size=18, max_diff=0.02, avg_diff=0.01, check_between=False)
    self.check_prediction(size=19, max_diff=0.03, avg_diff=0.01, check_between=False)
    self.check_prediction(size=20, max_diff=0.02, avg_diff=0.01, check_between=False)

  def check_prediction(self, size, max_diff, avg_diff, check_between=True):
    train, test = self.set

    predictor = LinearCurvePredictor(burn_in=len(train), min_input_size=1)
    predictor._x = np.array([item[0] for item in train])
    predictor._y = np.array([item[1] for item in train])

    test_x = np.array([item[0][:size] for item in test])
    test_y = np.array([item[1] for item in test])

    total_error = 0
    for i in range(test_x.shape[0]):
      x = test_x[i]
      y = test_y[i]

      predicted, error = predictor.predict(x)
      diff = abs(predicted - y)

      if check_between:
        self.check_between(predicted-error, y, predicted+error)
      self.assertLess(diff, max_diff)
      total_error += diff

    avg_error = total_error / test_x.shape[0]
    self.assertLess(avg_error, avg_diff)

  def check_between(self, x, y, z):
    if x > z: x, z = z, x
    self.assertLess(x, y)
    self.assertLess(y, z)

  def read_curve_data(self, train_size, test_size):
    data = DefaultIO._load_dict('curve-test-data.xjson')
    self.assertTrue(data)

    x = data['x'][:train_size]
    y = data['y'][:train_size]
    test_x = data['x'][train_size:train_size+test_size]
    test_y = data['y'][train_size:train_size+test_size]

    return list(zip(x, y)), list(zip(test_x, test_y))


if __name__ == '__main__':
  unittest.main()
