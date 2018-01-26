#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import re
import zipfile

import hyperengine as hype


# Mirror:
# https://cs.fit.edu/~mmahoney/compression/text8.zip
def get_text8(path='temp-text8/data', url='http://mattmahoney.net/dc/text8.zip'):
  zip_path = hype.util.download_if_needed(url, path)
  with zipfile.ZipFile(zip_path) as file_:
    data = file_.read(file_.namelist()[0])
  return data.decode('ascii').split()

def get_sms_spam_raw(path='temp-sms-spam/data',
                     url='http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'):
  zip_path = hype.util.download_if_needed(url, path)
  with zipfile.ZipFile(zip_path) as file_:
    data = file_.read('SMSSpamCollection')
  text_data = data.decode(errors='ignore')
  text_data = text_data.encode('ascii', errors='ignore')
  text_data = text_data.decode().split('\n')
  return text_data

def clean_text(text):
  text = re.sub(r'([^\s\w]|_|[0-9])+', '', text)
  text = ' '.join(text.split())
  text = text.lower()
  return text

def get_sms_spam_data(path='temp-sms-spam/data'):
  data = get_sms_spam_raw(path)
  data = [line.split('\t') for line in data if len(line) >= 1]
  [labels, inputs] = [list(line) for line in zip(*data)]
  inputs = [clean_text(line) for line in inputs]
  return inputs, labels
