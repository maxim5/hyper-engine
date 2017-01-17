#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


class TensorflowModel(object):
  def build_graph(self):
    raise NotImplementedError()

  def params_num(self):
    raise NotImplementedError()

  def hyper_params(self):
    raise NotImplementedError()

  def feed_dict(self):
    raise NotImplementedError()
