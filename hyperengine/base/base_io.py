#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import os

import numpy as np

from .logging import *
from .util import str_to_dict, dict_to_str

class BaseIO(object):
  """
  The base class responsible for I/O operations
  """

  def __init__(self, **params):
    self.load_dir = params.get('load_dir')
    self.save_dir = params.get('save_dir')

  @staticmethod
  def _prepare(directory):
    if directory is None:
      return False
    if not os.path.exists(directory):
      os.makedirs(directory)
    return True

  @staticmethod
  def _load_dict(from_file):
    if not os.path.exists(from_file):
      return {}
    try:
      with open(from_file, 'r') as file_:
        line = file_.readline()
        return str_to_dict(line)
    except:
      return {}


class Serializable(object):
  """
  Represents a serializable object
  """

  def import_from(self, data):
    raise NotImplementedError()

  def _import_from_keys(self, data, keys, default_value):
    for key in keys:
      value = np.array(data.get(key, default_value))
      setattr(self, '_%s' % key, value)

  def export_to(self):
    raise NotImplementedError()

  def _export_keys_to(self, keys):
    return {key: getattr(self, '_%s' % key).tolist() for key in keys}


class DefaultIO(BaseIO):
  """
  A simple I/O classes and can save and load a serializable instance.
  """

  def __init__(self, serializable, filename, **params):
    super(DefaultIO, self).__init__(**params)
    self.serializable = serializable
    self.filename = filename

  def load(self):
    directory = self.load_dir
    if not directory is None:
      destination = os.path.join(directory, self.filename)
      if os.path.exists(destination):
        data = DefaultIO._load_dict(destination)
        debug('Loaded data: %s from %s' % (dict_to_str(data), destination))
        self.serializable.import_from(data)
        return
    self.serializable.import_from({})

  def save(self):
    directory = self.save_dir
    if not DefaultIO._prepare(directory):
      return
    destination = os.path.join(directory, self.filename)
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(self.serializable.export_to()))
      debug('Data saved to %s' % destination)
