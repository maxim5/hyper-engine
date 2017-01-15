#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import operator


class BaseNode(object):
  def __init__(self, name=None):
    super(BaseNode, self).__init__()
    self._domain_value = None
    self._name = name

  def value(self):
    return self._domain_value

  def name(self):
    return self._name

  def with_name(self, name):
    self._name = name
    return self

  def describe(self):
    return None

  def __add__(self, other):  return _op2(self, other, operator.add)
  def __radd__(self, other): return _op2(self, other, operator.add, rev=True)

  def __sub__(self, other):  return _op2(self, other, operator.sub)
  def __rsub__(self, other): return _op2(self, other, operator.sub, rev=True)

  def __mul__(self, other):  return _op2(self, other, operator.mul)
  def __rmul__(self, other): return _op2(self, other, operator.mul, rev=True)

  def __div__(self, other):  return _op2(self, other, operator.div)
  def __rdiv__(self, other): return _op2(self, other, operator.div, rev=True)

  def __mod__(self, other):  return _op2(self, other, operator.mod)
  def __rmod__(self, other): return _op2(self, other, operator.mod, rev=True)

  def __floordiv__(self, other):  return _op2(self, other, operator.floordiv)
  def __rfloordiv__(self, other): return _op2(self, other, operator.floordiv, rev=True)

  def __pow__(self, other):  return _op2(self, other, operator.pow)
  def __rpow__(self, other): return _op2(self, other, operator.pow, rev=True)

  def __and__(self, other):  return _op2(self, other, operator.and_)
  def __rand__(self, other):  return _op2(self, other, operator.and_, rev=True)

  def __or__(self, other):  return _op2(self, other, operator.or_)
  def __ror__(self, other):  return _op2(self, other, operator.or_, rev=True)

  def __xor__(self, other):  return _op2(self, other, operator.__xor__)
  def __rxor__(self, other):  return _op2(self, other, operator.__xor__, rev=True)

  def __lshift__(self, other):  return _op2(self, other, operator.lshift)
  def __rlshift__(self, other): return _op2(self, other, operator.lshift, rev=True)

  def __rshift__(self, other):  return _op2(self, other, operator.rshift)
  def __rrshift__(self, other): return _op2(self, other, operator.rshift, rev=True)

  def __neg__(self):  return _op1(self, operator.neg)
  def __pos__(self):  return _op1(self, operator.pos)
  def __abs__(self):  return _op1(self, operator.abs)
  def __invert__(self):  return _op1(self, operator.invert)


def _op1(this, _operator):
  return MergeNode(_operator, this)

def _op2(this, other, _operator, rev=False):
  if isinstance(other, BaseNode):
    if rev:
      return MergeNode(_operator, other, this)
    else:
      return MergeNode(_operator, this, other)

  if rev:
    return MergeNode(lambda x: _operator(other, x), this)
  else:
    return MergeNode(lambda x: _operator(x, other), this)


class AcceptsInputNode(BaseNode):
  def __init__(self):
    super(AcceptsInputNode, self).__init__()
    self._point = None

  def set_point(self, point):
    self._point = point
    self._domain_value = self.to_domain_value(point)

  def to_domain_value(self, point):
    raise NotImplementedError()


class UniformNode(AcceptsInputNode):
  def __init__(self, start=0.0, end=1.0):
    super(UniformNode, self).__init__()
    self._shift = min(start, end)
    self._scale = abs(end - start)
    self._describe = 'uniform(%f, %f)' % (start, end)

  def to_domain_value(self, point):
    return point * self._scale + self._shift

  def describe(self):
    return self._describe


class NonUniformNode(AcceptsInputNode):
  def __init__(self, ppf, **args):
    super(NonUniformNode, self).__init__()
    self._ppf = ppf
    self._args = args

  def to_domain_value(self, point):
    return self._ppf(point, **self._args)

  def describe(self):
    return func_to_str(self._ppf) or 'complex'

def func_to_str(func):
  try:
    if hasattr(func, 'im_self'):
      self = func.im_self
      if hasattr(self, '__class__'):
        self = self.__class__
      return self.__name__
    return func.__name__
  except:
    return None


class ChoiceNode(AcceptsInputNode):
  def __init__(self, *array):
    super(ChoiceNode, self).__init__()
    self._array = array

  def to_domain_value(self, point):
    index = int(point * len(self._array))
    return self._array[min(index, len(self._array) - 1)]

  def describe(self):
    return 'choice(%d)' % len(self._array)


class JointNode(BaseNode):
  def __init__(self, *children):
    super(JointNode, self).__init__()
    self._children = children


class MergeNode(JointNode):
  def __init__(self, function, *children):
    super(MergeNode, self).__init__(*children)
    assert callable(function)
    self.function = function

  def value(self):
    if self._domain_value is None:
      self._domain_value = self.function(*[child.value() for child in self._children])
    return self._domain_value


class MergeChoiceNode(JointNode, AcceptsInputNode):
  def __init__(self, *children):
    super(MergeChoiceNode, self).__init__(*children)

  def value(self):
    if self._domain_value is None and self._point is not None:
      values = [child.value() if isinstance(child, BaseNode) else child for child in self._children]
      index = int(self._point * len(values))
      self._domain_value = values[min(index, len(values) - 1)]
    return self._domain_value

  def to_domain_value(self, point):
    return None

  def describe(self):
    return 'choice(%d)' % len(self._children)
