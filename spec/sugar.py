#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import math
from scipy import stats

from nodes import *

def wrap(node, transform):
  if transform is not None:
    return MergeNode(transform, node)
  return node

def uniform(start=0.0, end=1.0, transform=None, name=None):
  node = UniformNode(start, end).with_name(name)
  return wrap(node, transform)

def normal(mean=0.0, stdev=1.0, name=None):
  return NonUniformNode(ppf=stats.norm.ppf, loc=mean, scale=stdev).with_name(name)

def choice(array, transform=None, name=None):
  if not [item for item in array if isinstance(item, BaseNode)]:
    node = ChoiceNode(*array).with_name(name)
  else:
    node = MergeChoiceNode(*array).with_name(name)
  return wrap(node, transform)

def merge(nodes, function, name=None):
  if callable(nodes) and not callable(function):
    nodes, function = function, nodes
  if isinstance(nodes, BaseNode):
    nodes = [nodes]
  return MergeNode(function, *nodes).with_name(name)

def random_bit():
  return choice([0, 1])

def random_int(n):
  return choice(range(n))

def exp(node): return merge([node], math.exp)
def expm1(node): return merge([node], math.expm1)
def frexp(node): return merge([node], math.frexp)
def ldexp(node, i): return merge([node], lambda x: math.ldexp(x, i))

def sqrt(node): return merge([node], math.sqrt)
def pow(a, b): return a ** b

def log(node, base=None): return merge([node], lambda x: math.log(x, base))
def log1p(node): return merge([node], math.log1p)
def log10(node): return merge([node], math.log10)

def sin(node): return merge([node], math.sin)
def cos(node): return merge([node], math.cos)
def tan(node): return merge([node], math.tan)

def sinh(node): return merge([node], math.sinh)
def cosh(node): return merge([node], math.cosh)
def tanh(node): return merge([node], math.tanh)

def asin(node): return merge([node], math.asin)
def acos(node): return merge([node], math.acos)
def atan(node): return merge([node], math.atan)
def atan2(node): return merge([node], math.atan2)

def asinh(node): return merge([node], math.asinh)
def acosh(node): return merge([node], math.acosh)
def atanh(node): return merge([node], math.atanh)

def min_(*array):
  nodes = [item for item in array if isinstance(item, BaseNode)]
  if len(nodes) == 0:
    return min(*array) if len(array) > 1 else array[0]
  node = merge(nodes, min) if len(nodes) > 1 else nodes[0]

  rest = [item for item in array if not isinstance(item, BaseNode)]
  if rest:
    node = merge([node], lambda x: min(x, *rest))
  return node

def max_(*array):
  nodes = [item for item in array if isinstance(item, BaseNode)]
  if len(nodes) == 0:
    return max(*array) if len(array) > 1 else array[0]
  node = merge(nodes, max) if len(nodes) > 1 else nodes[0]

  rest = [item for item in array if not isinstance(item, BaseNode)]
  if rest:
    node = merge([node], lambda x: max(x, *rest))
  return node
