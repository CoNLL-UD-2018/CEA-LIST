#%reset -f
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import codecs
from argparse import ArgumentParser

from parser import Configurable
from parser import Network

# TODO make the pretrained vocab names a list given to TokenVocab
#***************************************************************
# Set up the argparser
argparser = ArgumentParser('Network')
argparser.add_argument('--save_dir', required=True, default='mymodel/')
subparsers = argparser.add_subparsers()
section_names = set()
# --section_name opt1=value1 opt2=value2 opt3=value3
with codecs.open(os.path.dirname(os.path.realpath(__file__)) + '/config/defaults.cfg') as f:
  section_regex = re.compile('\[(.*)\]')
  for line in f:
    match = section_regex.match(line)
    if match:
      section_names.add(match.group(1).lower().replace(' ', '_'))
#===============================================================
# Train
#---------------------------------------------------------------
def train(save_dir, **kwargs):
  """"""
  
  load = kwargs.pop('load')
  try:
    if not load and os.path.isdir(save_dir):
      print('Save directory already exists. Continuing..')
      #sys.exit(0)
      #raw_input('Save directory already exists. Press <Enter> to continue or <Ctrl-c> to abort.')
      #if os.path.isfile(os.path.join(save_dir, 'config.cfg')):
      #  os.remove(os.path.join(save_dir, 'config.cfg'))
  except KeyboardInterrupt:
    sys.exit(0)
  
  #print (kwargs)
  #print ("train files:",kwargs['train_files'])
  
  print ("initializing")
  network = Network(train=True,**kwargs)
  print ("initialized")
  network.train(load=load)
  return
#---------------------------------------------------------------

train_parser = subparsers.add_parser('train')
train_parser.set_defaults(action=train)
train_parser.add_argument('--load', action='store_true')
train_parser.add_argument('--config_file')
for section_name in section_names:
  train_parser.add_argument('--'+section_name, nargs='+')

#===============================================================
# Parse
#---------------------------------------------------------------
def parse(save_dir, **kwargs):
  """"""
  
  kwargs['config_file'] = os.path.join(save_dir, 'config.cfg')
  files = kwargs.pop('files')
  output_file = kwargs.pop('output_file', None)
  output_dir = kwargs.pop('output_dir', None)
  if len(files) > 1 and output_file is not None:
    raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
  kwargs['is_evaluation'] = True
  network = Network(**kwargs)
  network.parse(files, output_file=output_file, output_dir=output_dir)
  return
#---------------------------------------------------------------

parse_parser = subparsers.add_parser('parse')
parse_parser.set_defaults(action=parse)
parse_parser.add_argument('files', nargs='+')
for section_name in section_names:
  parse_parser.add_argument('--'+section_name, nargs='+')
parse_parser.add_argument('--output_file')
parse_parser.add_argument('--output_dir')

#===============================================================
# Tagger
#---------------------------------------------------------------
def tag(save_dir, **kwargs):
  """"""
  
  kwargs['config_file'] = os.path.join(save_dir, 'config.cfg')
  files = kwargs.pop('files')
  output_file = kwargs.pop('output_file', None)
  output_dir = kwargs.pop('output_dir', None)
  if len(files) > 1 and output_file is not None:
    raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
  kwargs['is_evaluation'] = True
  network = Network(**kwargs)
  network.tag(files, output_file=output_file, output_dir=output_dir)
  return
#---------------------------------------------------------------

parse_parser = subparsers.add_parser('tag')
parse_parser.set_defaults(action=tag)
parse_parser.add_argument('files', nargs='+')
for section_name in section_names:
  parse_parser.add_argument('--'+section_name, nargs='+')
parse_parser.add_argument('--output_file')
parse_parser.add_argument('--output_dir')

#===============================================================
# Freeze graph
#---------------------------------------------------------------
def test_frozen_graph(save_dir, **kwargs):
  """"""
  
  kwargs['config_file'] = os.path.join(save_dir, 'config.cfg')
  files = kwargs.pop('files')
  output_file = kwargs.pop('output_file', None)
  if len(files) > 1 and output_file is not None:
    raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
  kwargs['is_evaluation'] = True
  network = Network(**kwargs)
  network.test_frozen_graph(files,output_file=output_file)
  return
#---------------------------------------------------------------

parse_parser = subparsers.add_parser('test_frozen')
parse_parser.set_defaults(action=test_frozen_graph)
parse_parser.add_argument('files', nargs='+')
for section_name in section_names:
  parse_parser.add_argument('--'+section_name, nargs='+')
parse_parser.add_argument('--output_file')

#***************************************************************
# Parse the arguments

kwargs = vars(argparser.parse_args()) #"  --save_dir=mymodel_debug train  --config_file=config/config.cfg --pretrained_vocab max_rank=5000".split()
kwargs["load"] = True
action = kwargs.pop('action')
save_dir = kwargs.pop('save_dir')
kwargs = {key: value for key, value in kwargs.iteritems() if value is not None}
for section, values in kwargs.iteritems():
  if section in section_names:
    values = [value.split('=', 1) for value in values]
    kwargs[section] = {opt: value for opt, value in values}
if 'default' not in kwargs:
  kwargs['default'] = {}
kwargs['default']['save_dir'] = save_dir
action(save_dir, **kwargs)  
