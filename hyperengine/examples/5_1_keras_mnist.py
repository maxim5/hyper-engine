#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import hyperengine as hype


num_classes = 10
img_rows, img_cols = 28, 28

def prepare_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  return hype.Data(train=hype.DataSet(x_train, y_train),
                   validation=hype.DataSet(x_test, y_test),
                   test=hype.DataSet(x_test, y_test))

def build_model(params):
  model = Sequential()
  model.add(Conv2D(params.conv[0].size,
                   kernel_size=params.conv[0].kernel,
                   activation=params.conv[0].activation,
                   input_shape=(img_rows, img_cols, 1)))
  model.add(Conv2D(params.conv[1].size,
                   kernel_size=params.conv[1].kernel,
                   activation=params.conv[1].activation))
  model.add(MaxPooling2D(pool_size=params.pooling.size))
  model.add(Dropout(params.pooling.dropout))
  model.add(Flatten())
  model.add(Dense(params.dense.size, activation=params.dense.activation))
  model.add(Dropout(params.dense.dropout))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(lr=params.learning_rate),
                metrics=['accuracy'])
  return model


data = prepare_data()

def solver_generator(params):
  solver_params = {
    'batch_size': 2000,
    'epochs': 5,
  }
  model = build_model(params)
  solver = hype.KerasSolver(model, data=data, hyper_params=params, **solver_params)
  return solver


hyper_params_spec = hype.spec.new(
  learning_rate = 10**hype.spec.uniform(-1, -3),
  conv = [
    dict(
      size = 32,
      kernel = (3, 3),
      activation = 'relu',
    ),
    dict(
      size = 64,
      kernel = (3, 3),
      activation = 'relu',
    ),
  ],
  pooling = dict(
    size = (2, 2),
    dropout = 0.25,
  ),
  dense = dict(
    size = 128,
    activation = 'relu',
    dropout = 0.5,
  )
)
strategy_params = {
  'io_load_dir': 'temp-mnist/example-5-1',
  'io_save_dir': 'temp-mnist/example-5-1',
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
