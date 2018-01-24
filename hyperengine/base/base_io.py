#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import datetime
import os

import numpy as np

from .logging import *
from .util import str_to_dict, smart_str, random_id


class BaseIO(object):
  """
  The base class responsible for I/O operations
  """

  def __init__(self, **params):
    self.load_dir = params.get('load_dir')
    self.save_dir = _format_path(params.get('save_dir'))

  @staticmethod
  def _prepare(directory):
    if directory is None:
      return False
    if not os.path.exists(directory):
      vlog('Creating a directory ', directory)
      os.makedirs(directory)
    return True

  @staticmethod
  def _load_dict(from_file):
    if not os.path.exists(from_file):
      debug('Cannot load a dict. File does not exist: ', from_file)
      return {}
    try:
      with open(from_file, 'r') as file_:
        line = file_.readline()
        return str_to_dict(line)
    except BaseException as e:
      warn('Cannot load a dict. Error: ', e.message)
      return {}


class Serializable(object):
  """
  Represents a serializable object: can export own state to the dictionary and import back.
  """

  def import_from(self, data):
    """
    Imports the object state from the object `data`, usually a dictionary.
    """
    raise NotImplementedError()

  def _import_from_keys(self, data, keys, default_value):
    for key in keys:
      value = np.array(data.get(key, default_value))
      setattr(self, '_%s' % key, value)

  def export_to(self):
    """
    Exports the object to the dictionary.
    """
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
        if is_debug_logged():
          debug('Loaded data: %s from %s' % (smart_str(data), destination))
        self.serializable.import_from(data)
        return
    self.serializable.import_from({})

  def save(self):
    directory = self.save_dir
    if not DefaultIO._prepare(directory):
      return
    destination = os.path.join(directory, self.filename)
    with open(destination, 'w') as file_:
      data = smart_str(self.serializable.export_to())
      vlog('Exported data:', data)
      file_.write(data)
      debug('Data saved to:', destination)


def _format_path(directory):
  if directory and '{' in directory and '}' in directory:
    now = datetime.datetime.now()
    return directory.format(date=now.strftime('%Y-%m-%d'),
                            time=now.strftime('%H-%M-%S'),
                            random_id=random_id())
  return directory
