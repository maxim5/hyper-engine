#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import copy
import numbers
from six import iteritems

import numpy as np

from .nodes import BaseNode, AcceptsInputNode, JointNode


class ParsedSpec(object):
  """
  Responsible for parsing the spec and constructing a tree.
  When the spec is parsed, it can transform values from multi-dimensional [0,1] hypercube to domain values.
  """
  def __init__(self, spec):
    self._spec = spec
    self._input_nodes = {}
    self._traverse_nodes(spec)

  def size(self):
    return len(self._input_nodes)

  def instantiate(self, points):
    assert len(points) == self.size()
    for node, index in iteritems(self._input_nodes):
      node.set_point(points[index])

    spec_copy = copy.deepcopy(self._spec)
    spec_copy = self._traverse_and_replace(spec_copy)
    return spec_copy

  def get_names(self):
    return {index: node.name() for node, index in iteritems(self._input_nodes)}

  def _traverse_nodes(self, spec):
    self._visited = set()
    self._traverse_nodes_recursive(spec)
    self._visited = None

  def _visit(self, obj):
    if isinstance(obj, object) and hasattr(obj, '__dict__'):
      id_ = id(obj)
      if id_ in self._visited:
        return True
      self._visited.add(id_)
    return False

  def _traverse_nodes_recursive(self, spec, *path):
    if self._visit(spec):
      return

    if isinstance(spec, BaseNode):
      node = spec

      if isinstance(node, JointNode):
        for i, child in enumerate(node._children):
          self._traverse_nodes_recursive(child, *path)

      if isinstance(node, AcceptsInputNode) and not node in self._input_nodes:
        index = len(self._input_nodes)
        self._input_nodes[node] = index
        self._set_name(node, path)
      return

    if isinstance(spec, numbers.Number):
      return

    if isinstance(spec, dict):
      for key, value in iteritems(spec):
        self._traverse_nodes_recursive(value, key, *path)
      return

    if isinstance(spec, list) or isinstance(spec, tuple):
      for i, item in enumerate(spec):
        self._traverse_nodes_recursive(item, *path)
      return

    if isinstance(spec, object) and hasattr(spec, '__dict__'):
      for key, value in iteritems(spec.__dict__):
        if not (key.startswith('__') and key.endswith('__')):
          self._traverse_nodes_recursive(value, key, *path)
      return

  def _set_name(self, node, path):
    if node._name is None:
      name = '-'.join([str(i) for i in reversed(path)])
      describe = node.describe()
      if describe:
        name = name + '-' + describe
      node._name = name

  def _traverse_and_replace(self, spec_copy):
    self._visited = set()
    spec_copy = self._traverse_and_replace_recursive(spec_copy)
    self._visited = None
    return spec_copy

  def _traverse_and_replace_recursive(self, spec_copy):
    if isinstance(spec_copy, BaseNode):
      return spec_copy.value()

    if isinstance(spec_copy, numbers.Number):
      return spec_copy

    if self._visit(spec_copy):
      return spec_copy

    if isinstance(spec_copy, dict):
      for key, value in iteritems(spec_copy):
        spec_copy[key] = self._traverse_and_replace_recursive(spec_copy[key])
      return spec_copy

    if isinstance(spec_copy, list) or isinstance(spec_copy, tuple):
      replaced = [self._traverse_and_replace_recursive(item_copy) for item_copy in spec_copy]
      if isinstance(spec_copy, tuple):
        replaced = tuple(replaced)
      return replaced

    if isinstance(spec_copy, object) and hasattr(spec_copy, '__dict__'):
      for key, value in iteritems(spec_copy.__dict__):
        if not(key.startswith('__') and key.endswith('__')):
          setattr(spec_copy, key, self._traverse_and_replace_recursive(value))
      return spec_copy

    return spec_copy


def get_instance(spec):
  parsed = spec if isinstance(spec, ParsedSpec) else ParsedSpec(spec)
  points = np.random.uniform(0, 1, size=(parsed.size(),))
  return parsed.instantiate(points)
