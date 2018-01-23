#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import unittest

import numpy as np

import hyperengine as hype
from hyperengine.base import NamedDict


class NamedDictTest(unittest.TestCase):
  def test_embedded_dict(self):
    spec = hype.spec.new({
      'foo': {
        'bar': {
          'baz': 999
        },
        'baz': []
      }
    })

    instance = self._instantiate(spec)
    self.assertEqual(repr(instance), "{'foo': {'bar': {'baz': 999}, 'baz': []}}")
    self.assertItemsEqual(instance.foo.keys(), ['bar', 'baz'])

    self.assertTrue('foo' in instance)
    self.assertTrue('bar' in instance.foo)
    self.assertTrue('baz' in instance.foo.bar)

    self.assertEqual(type(instance), NamedDict)
    self.assertEqual(type(instance.foo), NamedDict)

    self.assertEqual(instance.foo.bar.baz, 999)
    self.assertEqual(instance.foo.baz, [])


  def test_embedded_list(self):
    spec = hype.spec.new(
      value = [[1], [[2]], (3, 4), {'foo': 5}],
    )

    instance = self._instantiate(spec)
    self.assertEqual(type(instance), NamedDict)
    self.assertEqual(type(instance.value), list)
    self.assertEqual(type(instance.value[3]), NamedDict)
    self.assertEqual(repr(instance), "{'value': [[1], [[2]], [3, 4], {'foo': 5}]}")
    self.assertEqual(instance.value[3].foo, 5)


  def test_dict_inside_list(self):
    spec = hype.spec.new(
      foo = [
        { 'bar': 0 },
        { 'bar': 1 },
        { 'bar': 2 },
      ]
    )

    instance = self._instantiate(spec)
    self.assertEqual(type(instance), NamedDict)
    self.assertEqual(type(instance.foo), list)
    self.assertEqual(type(instance.foo[0]), NamedDict)
    self.assertEqual(instance.foo[0].bar, 0)
    self.assertEqual(instance.foo[1].bar, 1)
    self.assertEqual(instance.foo[2].bar, 2)


  def test_real(self):
    hyper_params_spec = hype.spec.new(
      learning_rate=10 ** hype.spec.uniform(-2, -3),
      conv=hype.spec.new(
        filters=[hype.spec.choice([20, 32, 48]), hype.spec.choice([64, 96, 128])],
        residual=hype.spec.random_bit(),
      ),
      dropout=hype.spec.uniform(0.5, 0.9),
    )

    instance = self._instantiate(hyper_params_spec)
    self.assertEqual(repr(instance),
                     "{'conv': {'filters': [20, 64], 'residual': 0}, 'dropout': 0.500000, 'learning_rate': 0.001000}")
    self.assertItemsEqual(instance.keys(), ['learning_rate', 'conv', 'dropout'])

    self.assertEqual(type(instance), NamedDict)
    self.assertEqual(type(instance.conv), NamedDict)

    self.assertEqual(instance.learning_rate, 0.001)
    self.assertEqual(instance['learning_rate'], 0.001)
    self.assertEqual(instance.get('learning_rate'), 0.001)

    self.assertEqual(instance.get('foo'), None)
    self.assertEqual(instance.get('foo', 'bar'), 'bar')

    self.assertItemsEqual(instance.conv.keys(), ['filters', 'residual'])
    self.assertEqual(instance.conv.filters, [20, 64])
    self.assertEqual(instance.conv.filters[0], 20)
    self.assertEqual(instance.conv.filters[1], 64)


  def _instantiate(self, spec):
    parsed = hype.spec.ParsedSpec(spec)
    points = np.zeros([parsed.size()])
    instance = parsed.instantiate(points)
    return instance
