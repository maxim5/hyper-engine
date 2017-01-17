#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import os

from ..base import *


class ModelIO(BaseIO):
  def __init__(self, **params):
    super(ModelIO, self).__init__(**params)
    self.data_saver = params.get('data_saver')

  def save_results(self, results, directory=None):
    directory = directory or self.save_dir
    if not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'results.xjson')
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(results))
      debug('Results saved to %s' % destination)

  def load_results(self, directory, log_level):
    if directory is None:
      return

    destination = os.path.join(directory, 'results.xjson')
    if os.path.exists(destination):
      results = ModelIO._load_dict(destination)
      log_at_level(log_level, 'Loaded results: %s from %s' % (dict_to_str(results), destination))
      return results

  def save_hyper_params(self, hyper_params, directory=None):
    directory = directory or self.save_dir
    if not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'hyper_params.xjson')
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(hyper_params))
      debug('Hyper parameters saved to %s' % destination)

  def load_hyper_params(self, directory=None):
    directory = directory or self.load_dir
    if directory is None:
      return

    hyper_params = ModelIO._load_dict(os.path.join(directory, 'hyper_params.xjson'))
    if hyper_params:
      info('Loaded hyper-params: %s' % dict_to_str(hyper_params))
      return hyper_params

  def save_data(self, data, directory=None):
    directory = directory or self.save_dir
    if self.data_saver is None or data is None or not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'misclassified')
    actual_destination = call(self.data_saver, data, destination)
    if actual_destination:
      debug('Misclassified data saved to %s' % actual_destination)
    else:
      warn('Data saver can not be not called or returns None')
