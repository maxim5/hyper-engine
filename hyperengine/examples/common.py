#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import os
import re
import sys
import zipfile

from six.moves import urllib

import hyperengine as hype


def download_if_needed(url, path, filename=None):
  if not os.path.exists(path):
    os.makedirs(path)
  filename = filename or os.path.basename(url)
  full_path = os.path.join(path, filename)
  if not os.path.exists(full_path):
    hype.util.info('Downloading %s, please wait...' % filename)
    result_path, _ = urllib.request.urlretrieve(url, full_path, report_hook)
    stat = os.stat(result_path)
    hype.util.info('Successfully downloaded', filename, stat.st_size, 'b')
    return result_path
  else:
    hype.util.debug('Already downloaded:', full_path)
  return full_path

def report_hook(block_num, block_size, total_size):
  read_so_far = block_num * block_size
  if total_size > 0:
    percent = read_so_far * 1e2 / total_size
    s = '\r%5.1f%% %*d / %d' % (percent, len(str(total_size)), read_so_far, total_size)
    sys.stderr.write(s)
    if read_so_far >= total_size:  # near the end
      sys.stderr.write('\n')
  else:  # total size is unknown
    sys.stderr.write('read %d\n' % (read_so_far,))


########################################################################################################################
# Data sets
########################################################################################################################


# Mirror:
# https://cs.fit.edu/~mmahoney/compression/text8.zip
def get_text8(path='temp-text8/data', url='http://mattmahoney.net/dc/text8.zip'):
  zip_path = download_if_needed(url, path)
  with zipfile.ZipFile(zip_path) as file_:
    data = file_.read(file_.namelist()[0])
  return data.decode('ascii').split()

def get_sms_spam_raw(path='temp-sms-spam/data',
                     url='http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'):
  zip_path = download_if_needed(url, path)
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
