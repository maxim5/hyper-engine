#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import functools

def graph_vars(graph):
  for n in graph.as_graph_def().node:
    element = graph.as_graph_element(n.name)
    if element.type == 'Variable' or element.type == 'VariableV2':
      yield element

def get_total_dim(element):
  # See TensorShapeProto
  shape = element.get_attr('shape')
  dims = shape.dim
  return functools.reduce(lambda x, y: x * y, [dim.size for dim in dims], 1)
