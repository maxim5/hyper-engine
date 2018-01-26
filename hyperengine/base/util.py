#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import ast
import collections
import numbers
import os
import random
import string
import sys
from six.moves import urllib

import numpy as np


def smart_str(val):
  if type(val) in [float, np.float32, np.float64] and val:
    return '%.6f' % val if abs(val) > 1e-6 else '%e' % val
  if type(val) == dict:
    return '{%s}' % ', '.join(['%s: %s' % (repr(k), smart_str(val[k])) for k in sorted(val.keys())])
  if type(val) in [list, tuple]:
    return '[%s]' % ', '.join(['%s' % smart_str(i) for i in val])
  return repr(val)

def str_to_dict(s):
  return ast.literal_eval(s)

def zip_longest(list1, list2):
  len1 = len(list1)
  len2 = len(list2)
  for i in xrange(max(len1, len2)):
    yield (list1[i % len1], list2[i % len2])

def deep_update(dict_, upd):
  for key, value in upd.iteritems():
    if isinstance(value, collections.Mapping):
      recursive = deep_update(dict_.get(key, {}), value)
      dict_[key] = recursive
    else:
      dict_[key] = upd[key]
  return dict_

def mini_batch(total, size):
  return zip(range(0, total, size),
             range(size, total + size, size))

def random_id(size=6, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in xrange(size))

def safe_concat(list_):
  list_ = [i for i in list_ if i is not None]
  if len(list_) == 0:
    return None
  if type(list_[0]) == np.ndarray:
    return np.concatenate(list_)
  return list_

def call(obj, *args):
  if callable(obj):
    return obj(*args)

  apply = getattr(obj, 'apply', None)
  if callable(apply):
    return apply(*args)

def slice_dict(d, key_prefix):
  return {key[len(key_prefix):]: value for key, value in d.iteritems() if key.startswith(key_prefix)}

def as_function(val, presets, default=None):
  if callable(val):
    return val

  preset = presets.get(val, default)
  if preset is not None:
    return preset

  raise ValueError('Value is not recognized: ', val)

def as_numeric_function(val, presets, default=None):
  if isinstance(val, numbers.Number):
    def const(*_):
      return val
    return const

  return as_function(val, presets, default)


def download_if_needed(url, path, filename=None):
  from .logging import info, debug

  if not os.path.exists(path):
    os.makedirs(path)
  filename = filename or os.path.basename(url)
  full_path = os.path.join(path, filename)
  if not os.path.exists(full_path):
    info('Downloading %s, please wait...' % filename)
    result_path, _ = urllib.request.urlretrieve(url, full_path, _report_hook)
    stat = os.stat(result_path)
    info('Successfully downloaded "%s" (%d Kb)' % (filename, stat.st_size / 1024))
    return result_path
  else:
    debug('Already downloaded:', full_path)
  return full_path


def _report_hook(block_num, block_size, total_size):
  read_so_far = block_num * block_size
  if total_size > 0:
    percent = read_so_far * 1e2 / total_size
    s = '\r%5.1f%% %*d / %d' % (percent, len(str(total_size)), read_so_far, total_size)
    sys.stdout.write(s)
    if read_so_far >= total_size:  # near the end
      sys.stdout.write('\n')
  else:  # total size is unknown
    sys.stdout.write('read %d\n' % (read_so_far,))
