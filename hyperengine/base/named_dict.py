#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


from .util import smart_str

# Inspired by https://stackoverflow.com/questions/1305532/convert-python-dict-to-object
class NamedDict(object):
  def __init__(self, input):
    for k, v in input.items():
      if isinstance(v, (list, tuple)):
        setattr(self, k, [NamedDict(x) if isinstance(x, dict) else x for x in v])
      else:
        setattr(self, k, NamedDict(v) if isinstance(v, dict) else v)

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def get(self, key, default=None):
    return self.__dict__.get(key, default)

  def __getitem__(self, key):
    return self.__dict__[key]

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return smart_str(self.__dict__)
