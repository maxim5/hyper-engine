#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


class BaseModel(object):
  def build_graph(self):
    pass

  def params_num(self):
    pass

  def hyper_params(self):
    pass

  def feed_dict(self):
    pass
