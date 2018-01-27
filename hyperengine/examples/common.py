#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import os
import pickle
import re
import sys
import tarfile
import zipfile

import numpy as np

import hyperengine as hype


########################################################################################################################
# CIFAR-10
########################################################################################################################


def get_cifar10(path='temp-cifar10/data',
                url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                one_hot=False,
                validation_size=None):
  tar_file = hype.util.download_if_needed(url, path)
  untar(tar_file)
  path = os.path.join(path, 'cifar-10-batches-py')

  x_train = []
  y_train = []
  for i in range(1, 6):
    batch_path = os.path.join(path, 'data_batch_%d' % i)
    data, labels = load_cifar10_batch(batch_path)
    if i == 1:
      x_train = data
      y_train = labels
    else:
      x_train = np.concatenate([x_train, data], axis=0)
      y_train = np.concatenate([y_train, labels], axis=0)

  batch_path = os.path.join(path, 'test_batch')
  x_test, y_test = load_cifar10_batch(batch_path)

  x_train = np.dstack((x_train[:, :1024], x_train[:, 1024:2048],
                       x_train[:, 2048:])) / 255.
  x_train = np.reshape(x_train, [-1, 32, 32, 3])
  x_test = np.dstack((x_test[:, :1024], x_test[:, 1024:2048],
                      x_test[:, 2048:])) / 255.
  x_test = np.reshape(x_test, [-1, 32, 32, 3])

  if one_hot:
    y_train = encode_one_hot(y_train, 10)
    y_test = encode_one_hot(y_test, 10)

  if validation_size is not None:
    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]
    x_train = x_train[validation_size:]
    y_train = y_train[validation_size:]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

  return (x_train, y_train), (x_test, y_test)


def load_cifar10_batch(filename):
  with open(filename, 'rb') as file_:
    if sys.version_info > (3, 0):
      d = pickle.load(file_, encoding='latin1')   # Python3
    else:
      d = pickle.load(file_)                      # Python2
  return d['data'], d['labels']


def untar(filename):
  if filename.endswith('tar.gz'):
    tar = tarfile.open(filename)
    tar.extractall(path=os.path.dirname(filename))
    tar.close()
    hype.util.debug('File "%s" extracted in current directory' % filename)
  else:
    hype.util.warn('Not a tar.gz file: "%s"' % filename)


def encode_one_hot(y, num_classes=None):
  """
  Convert class vector (integers from 0 to num_classes) to binary class matrix.
  Arguments:
      y: `array`. Class vector to convert.
      num_classes: `int`. Total number of classes.
  """
  y = np.asarray(y, dtype='int32')
  if not num_classes:
    num_classes = np.max(y) + 1
  one_hot = np.zeros((len(y), num_classes))
  one_hot[np.arange(len(y)), y] = 1.0
  return one_hot


########################################################################################################################
# Text8 (Wikipedia text corpus)
########################################################################################################################


# Mirror:
# https://cs.fit.edu/~mmahoney/compression/text8.zip
def get_text8(path='temp-text8/data', url='http://mattmahoney.net/dc/text8.zip'):
  zip_path = hype.util.download_if_needed(url, path)
  with zipfile.ZipFile(zip_path) as file_:
    data = file_.read(file_.namelist()[0])
  return data.decode('ascii').split()


########################################################################################################################
# SMS spam collection
########################################################################################################################


def get_sms_spam_raw(path='temp-sms-spam/data',
                     url='http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'):
  zip_path = hype.util.download_if_needed(url, path)
  with zipfile.ZipFile(zip_path) as file_:
    data = file_.read('SMSSpamCollection')
  text_data = data.decode(errors='ignore')
  text_data = text_data.encode('ascii', errors='ignore')
  text_data = text_data.decode().split('\n')
  return text_data

def clean_text(text):
  text = re.sub(r'([^\s\w]|_|[0-9])+', '', text)
  text = ' '.join(text.split())
  text = text.lower()
  return text

def get_sms_spam_data(path='temp-sms-spam/data'):
  data = get_sms_spam_raw(path)
  data = [line.split('\t') for line in data if len(line) >= 1]
  [labels, inputs] = [list(line) for line in zip(*data)]
  inputs = [clean_text(line) for line in inputs]
  return inputs, labels
