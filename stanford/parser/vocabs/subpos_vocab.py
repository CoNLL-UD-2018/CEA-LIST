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

import os
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.vocabs import TokenVocab
from parser import Multibucket
from parser.misc.bucketer import Bucketer


#***************************************************************
class SubposVocab(TokenVocab):
  """"""
  
  #=============================================================
  def __init__(self, name, token_vocab, *args, **kwargs):
    """"""
    
    recount = kwargs.pop('recount', False)
    initialize_zero = kwargs.pop('initialize_zero', True)
    #print(token_vocab,',',name)
    super(TokenVocab, self).__init__(*args, **kwargs)
    self._name = name;
    
    self.separator = token_vocab.separator
    self._token_vocab = token_vocab
    self._token_counts = Counter()
    self._category = self.name[4:]
    
    
    if recount:
      self.count()
    else:
      print(os.path.join(self.save_dir, self.name+'.txt'), " exist ?")
      if os.path.exists(os.path.join(self.save_dir, self.name+'.txt')):
        self.load()
      else:
        self.count()
        self.dump()
    self.index_vocab()
    
    embed_dims = [len(self), self.embed_size]
    if initialize_zero:
      self.embeddings = np.zeros(embed_dims)
    else:
      self.embeddings = np.random.randn(*embed_dims)
    return
  

  
  #=============================================================
  def count(self):
    """"""
    
    special_tokens = set(self.token_vocab.special_tokens)
    category = self.name[4:]
    #One subpos category == 'all' means it'll contain every categories of subpos
    if category != 'all':
        for token in self.token_vocab.counts:
          for subpos in token.split(self.separator):
              subpos = subpos.replace('[','')
              subpos = subpos.replace(']','')
              if self.name[0:4] == 'feat':
                  if subpos.startswith(category+'='):
                      self.counts[subpos] += 1
                      self.token_counts[subpos] += self.token_vocab.counts[token]
              elif self.name[0:4] == 'xtag':
                  if subpos == category:
                      self.counts[subpos] += 1
                      self.token_counts[subpos] += self.token_vocab.counts[token]
    else:
        for token in self.token_vocab.counts:
          for subpos in [token]:
              self.counts[subpos] += 1
              self.token_counts[subpos] += self.token_vocab.counts[token]
        
    return
  
  #=============================================================
  def load(self):
    """"""
    
    train_file = os.path.join(self.save_dir, self.name+'.txt')
    with codecs.open(train_file, encoding='utf-8') as f:
      for line_num, line in enumerate(f):
        try:
          line = line.rstrip()
          if line:
            line = line.split('\t')
            token, count, token_count = line
            self.counts[token] = int(count)
            self.token_counts[token] = int(token_count)
        except:
          raise ValueError('File %s is misformatted at line %d' % (train_file, line_num+1))
    return
  
  #=============================================================
  def dump(self):
    """"""
    
    with codecs.open(os.path.join(self.save_dir, self.name+'.txt'), 'w', encoding='utf-8') as f:
      for token, count in self.sorted_counts(self._counts):
        f.write('%s\t%d\t%d\n' % (token, count, self.token_counts[token]))
    return
  
  #=============================================================
  def subpos_indices(self, token):
    """"""
        
    subposid = -1
    for k in token.split(self.separator):
        k = k.replace('[','')
        k = k.replace(']','')
        sub = self._str2idx.get(k)
        if sub != None:
            subposid = sub
            break
    if subposid == -1:
        subposid = self.UNK
    
    return subposid

  
  #=============================================================
  def index_tokens(self):
    """"""
    category = self.name[4:]
    if category != 'all':
        #Associe à chaque token un indice
        self._tok2idx = {token: self.subpos_indices(token) for token in self.counts}
    else:
        self._tok2idx = {token: self[token] for token in self.token_vocab.counts}
        
    return
  

  
  #=============================================================
  def index(self, token):
    category = self.name[4:]
    
    if category != 'all':
      for subpos in token.split(self.separator):
          if subpos in self._tok2idx:
              return self._tok2idx.get(subpos)
    else:
        if token in self._tok2idx:
            return self._tok2idx.get(token)
    return self.META_UNK
  
  #=============================================================
  
  @property
  def token_counts(self):
    return self._token_counts
  @property
  def token_vocab(self):
    return self._token_vocab
  @property
  def token_embed_size(self):
    return (self.token_vocab or self).embed_size
  @property
  def conll_idx(self):
    return self.token_vocab.conll_idx
  @property
  def tok2idx(self):
    return self._tok2idx
  @property
  def idx2tok(self):
    return self._idx2tok
  @property
  def name(self):
    if self._name is None:
      return self.get('name')
    else:
      return self._name
  @name.setter
  def name(self,name):
    self.name = name
  
  #=============================================================
  def __setattr__(self, name, value):
    if name == '_token_vocab':
      if self.cased is None:
        self._cased = value.cased
      elif self.cased != value.cased:
        cls = value.__class__
        value = cls.from_configurable(value,
                                      cased=self.cased,
                                      recount=True)
    super(SubposVocab, self).__setattr__(name, value)
    return

  def __getitem__(self, key):
    if isinstance(key, basestring) and self._category == 'all':
      return self._str2idx.get(key, self.UNK)
    elif isinstance(key, (int, long, np.int32, np.int64)):
      return self._idx2str.get(key, self.special_tokens[self.UNK])
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to BaseVocab.__getitem__ must be (iterable of) string or integer')
    return



  