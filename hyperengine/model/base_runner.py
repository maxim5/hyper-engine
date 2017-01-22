#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


class BaseRunner(object):
  """
  The runner represents a connecting layer between the solver and the machine learning model.
  Responsible for communicating with the model with a data batch: prepare, train, evaluate.
  """

  def build_model(self, **kwargs):
    """
    Builds and prepares a model. Method is not expected to return anything.
    """
    raise NotImplementedError()

  def run_batch(self, batch_x, batch_y):
    """
    Runs the training iteration for a batch of data. Method is not expected to return anything.
    """
    raise NotImplementedError()

  def evaluate(self, batch_x, batch_y):
    """
    Evaluates the test result for a batch of data. Method should return the dictionary that contains
    one (or all) of the following:
    - batch accuracy (key 'accuracy')
    - associated cost (key 'cost')
    - any other computed data (key 'data')
    """
    raise NotImplementedError()

  def describe(self):
    """
    Describes the model. Method should return the dictionary that contains one (or all) of the following:
    - model size (number of learnable parameters) (key 'model_size')
    - model memory requirements (key 'model_memory')
    - hyper parameters used (key 'hyper_parameters')
    """
    raise NotImplementedError()
