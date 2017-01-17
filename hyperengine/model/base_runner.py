#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


class BaseRunner(object):
  def build_model(self, **kwargs):
    pass

  def run_batch(self, batch_x, batch_y):
    pass

  def evaluate(self, batch_x, batch_y):
    return {}

  def describe(self):
    return {}
