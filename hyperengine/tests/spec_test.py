#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import six
import unittest

from hyperengine.spec import *


class SpecTest(unittest.TestCase):
  def test_zero_nodes(self):
    def check_zero_nodes(spec):
      parsed = ParsedSpec(spec)
      self.assertEqual(parsed.size(), 0)
      self.assertEqual(spec, parsed.instantiate([]))
    
    check_zero_nodes(1)
    check_zero_nodes([])
    check_zero_nodes([1, 2, 3])
    check_zero_nodes((1, 2, 3))
    check_zero_nodes({})
    check_zero_nodes({'a': 0, 'b': 1})
    check_zero_nodes({'a': [1, 2], 'b': {'key': (1, 2)}})


  def test_uniform(self):
    spec = uniform()
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.0, parsed.instantiate([0.0]))
    self.assertEqual(0.5, parsed.instantiate([0.5]))
    self.assertEqual(1.0, parsed.instantiate([1.0]))


  def test_uniform_rev(self):
    spec = uniform(4, 0)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.0, parsed.instantiate([0.0]))
    self.assertEqual(2.0, parsed.instantiate([0.5]))
    self.assertEqual(4.0, parsed.instantiate([1.0]))


  def test_uniform_negative(self):
    spec = uniform(-4, -2)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(-4.0, parsed.instantiate([0.0]))
    self.assertEqual(-3.0, parsed.instantiate([0.5]))
    self.assertEqual(-2.0, parsed.instantiate([1.0]))


  def test_uniform_negative_rev(self):
    spec = uniform(-2, -4)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(-4.0, parsed.instantiate([0.0]))
    self.assertEqual(-3.0, parsed.instantiate([0.5]))
    self.assertEqual(-2.0, parsed.instantiate([1.0]))


  def test_normal(self):
    spec = normal()
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertAlmostEqual(-1.0, parsed.instantiate([0.1587]), delta=0.001)
    self.assertAlmostEqual(-0.5, parsed.instantiate([0.3085]), delta=0.001)
    self.assertAlmostEqual( 0.0, parsed.instantiate([0.5000]), delta=0.001)
    self.assertAlmostEqual( 0.7, parsed.instantiate([0.7580]), delta=0.001)
    self.assertAlmostEqual( 0.9, parsed.instantiate([0.8159]), delta=0.001)


  def test_choice(self):
    spec = choice([10, 20, 30])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(10, parsed.instantiate([0.0]))
    self.assertEqual(20, parsed.instantiate([0.5]))
    self.assertEqual(30, parsed.instantiate([1.0]))


  def test_choice_str(self):
    spec = choice(['foo', 'bar'])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual('foo', parsed.instantiate([0.0]))
    self.assertEqual('bar', parsed.instantiate([1.0]))


  def test_merge(self):
    spec = merge([uniform(), uniform()], lambda x, y: x+y)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual(0.5, parsed.instantiate([0.0, 0.5]))
    self.assertEqual(1.5, parsed.instantiate([0.5, 1.0]))
    self.assertEqual(2.0, parsed.instantiate([1.0, 1.0]))


  def test_transform(self):
    spec = wrap(uniform(), lambda x: x*x)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.0, parsed.instantiate([0.0]))
    self.assertEqual(4.0, parsed.instantiate([2.0]))


  def test_transform_merge(self):
    spec = wrap(merge([uniform(), uniform()], lambda x, y: x+y), lambda x: x*x)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual(1.0, parsed.instantiate([0.0, 1.0]))
    self.assertEqual(4.0, parsed.instantiate([1.0, 1.0]))


  def test_duplicate_nodes_1(self):
    node = uniform()
    spec = merge([node, node, node], lambda x, y, z: x+y+z)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(3.0, parsed.instantiate([1.0]))
    self.assertEqual(9.0, parsed.instantiate([3.0]))


  def test_duplicate_nodes_2(self):
    node = uniform()
    spec = [[node, node]]
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual([[1.0, 1.0]], parsed.instantiate([1.0]))


  def test_duplicate_nodes_3(self):
    spec = [uniform()] * 3
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual([0.0, 0.0, 0.0], parsed.instantiate([0.0]))
    self.assertEqual([1.0, 1.0, 1.0], parsed.instantiate([1.0]))


  def test_merge_choice(self):
    spec = choice([uniform(0, 1), uniform(2, 3)])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual(0.0, parsed.instantiate([0.0, 0.0, 0.0]))
    self.assertEqual(1.0, parsed.instantiate([1.0, 0.0, 0.0]))
    self.assertEqual(2.0, parsed.instantiate([0.0, 0.0, 0.9]))
    self.assertEqual(3.0, parsed.instantiate([0.0, 1.0, 0.9]))


  def test_if_condition(self):
    def if_cond(switch, size, num):
      if switch > 0.5:
        return [size, num, num]
      return [size, num]

    spec = merge([uniform(0, 1), uniform(1, 2), uniform(2, 3)], if_cond)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)

    self.assertEqual([1, 2],    parsed.instantiate([0, 0, 0]))
    self.assertEqual([2, 3],    parsed.instantiate([0, 1, 1]))
    self.assertEqual([1, 2, 2], parsed.instantiate([1, 0, 0]))
    self.assertEqual([2, 3, 3], parsed.instantiate([1, 1, 1]))


  def test_object(self):
    class Dummy: pass
    dummy = Dummy
    dummy.value = uniform()
    dummy.foo = 'bar'
    dummy.ref = dummy

    spec = dummy
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)

    instance = parsed.instantiate([0])
    self.assertEqual(0, instance.value)
    self.assertEqual('bar', instance.foo)
    self.assertEqual(instance, instance.ref)


  def test_dict(self):
    spec = {1: uniform(), 2: choice(['foo', 'bar']), 3: merge(lambda x: -x, uniform())}
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual({1: 0.0, 2: 'foo', 3:  0.0}, parsed.instantiate([0, 0, 0]))
    self.assertEqual({1: 1.0, 2: 'bar', 3: -1.0}, parsed.instantiate([1, 1, 1]))


  def test_dict_deep_1(self):
    spec = {1: {'foo': uniform() } }
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)


  def test_dict_deep_2(self):
    spec = {'a': {'b': {'c': { 'd': uniform() } } } }
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)


  def test_math_operations_1(self):
    spec = uniform() + 1
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(2.0, parsed.instantiate([1.0]))


  def test_math_operations_2(self):
    spec = uniform() * (uniform() ** 2 + 1) / uniform()
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual(2.0, parsed.instantiate([1.0, 1.0, 1.0]))
    self.assertEqual(1.0, parsed.instantiate([0.5, 1.0, 1.0]))
    self.assertEqual(1.0, parsed.instantiate([0.5, 0.0, 0.5]))


  def test_math_operations_3(self):
    spec = 2 / (1 + uniform()) * (3 - uniform() + 4 ** uniform())
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual(6.0, parsed.instantiate([1.0, 1.0, 1.0]))


  def test_math_operations_4(self):
    spec = choice(['foo', 'bar']) + '-' + choice(['abc', 'def'])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual('foo-abc', parsed.instantiate([0.0, 0.0]))
    self.assertEqual('bar-def', parsed.instantiate([1.0, 1.0]))


  def test_min_1(self):
    spec = min(uniform(), uniform(), 0.5)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual(0.5, parsed.instantiate([1.0, 0.7]))
    self.assertEqual(0.5, parsed.instantiate([1.0, 0.5]))
    self.assertEqual(0.0, parsed.instantiate([0.0, 0.5]))


  def test_min_2(self):
    spec = min(uniform(), 0.8, 0.5)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.5, parsed.instantiate([1.0]))
    self.assertEqual(0.5, parsed.instantiate([0.5]))
    self.assertEqual(0.2, parsed.instantiate([0.2]))


  def test_min_3(self):
    spec = min(uniform(), uniform())
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual(0.5, parsed.instantiate([1.0, 0.5]))
    self.assertEqual(0.2, parsed.instantiate([0.2, 0.5]))


  def test_max_1(self):
    spec = max(0.5)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 0)
    self.assertEqual(0.5, parsed.instantiate([]))


  def test_max_2(self):
    spec = max(0.5, 1.0)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 0)
    self.assertEqual(1.0, parsed.instantiate([]))


  def test_max_3(self):
    spec = max(uniform())
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(1.0, parsed.instantiate([1.0]))
    self.assertEqual(0.0, parsed.instantiate([0.0]))


  def test_name_1(self):
    aaa = uniform()
    bbb = choice(['foo'])
    ccc = uniform(-1, 1)
    ddd = uniform()
    spec = {'aaa': aaa, 'bbb': bbb, 'ccc': ccc **2, 'ddd': [ddd, ddd]}

    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 4)
    self.assertTrue('aaa' in aaa.name())
    self.assertTrue('uniform' in aaa.name())
    self.assertTrue('bbb' in bbb.name())
    self.assertTrue('choice' in bbb.name())
    self.assertTrue('ccc' in ccc.name())
    self.assertTrue('uniform' in ccc.name())
    self.assertTrue('ddd' in ddd.name())
    self.assertTrue('uniform' in ddd.name())


  def test_name_2(self):
    norm_node = normal()
    choice_node = choice([uniform(), uniform(), uniform()])
    spec = {'a': {'b': {'c': { 'd': norm_node, 0: choice_node } } } }

    # stats.norm.ppf is an instance method in python 2
    expected_normal_name = 'norm_gen' if six.PY2 else 'ppf'

    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 5)
    self.assertTrue('a-b-c-d' in norm_node.name(), 'name=%s' % norm_node.name())
    self.assertTrue(expected_normal_name in norm_node.name(), 'name=%s' % norm_node.name())
    self.assertTrue('a-b-c-0' in choice_node.name(), 'name=%s' % choice_node.name())
    self.assertTrue('choice' in choice_node.name(), 'name=%s' % choice_node.name())
