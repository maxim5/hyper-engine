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
    Builds and prepares a model.
    """
    raise NotImplementedError()

  def run_batch(self, batch_x, batch_y):
    """
    Runs the training for a batch of data.
    """
    raise NotImplementedError()

  def evaluate(self, batch_x, batch_y):
    """
    Evaluates the test result for a batch of data.
    """
    raise NotImplementedError()

  def describe(self):
    """
    Describes the model.
    """
    raise NotImplementedError()
