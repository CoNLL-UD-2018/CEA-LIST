#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import gzip
import warnings
import base64
import hashlib
#try:
#  from backports import lzma
#except:
#  warnings.warn('Install backports.lzma for xz support')
from collections import Counter

import numpy as np
import tensorflow as tf

import parser.neural.linalg as linalg
from parser.vocabs.pretrained_vocab import PretrainedVocab
from parser.vocabs.base_vocab import BaseVocab

#***************************************************************
class TagRepVocab(BaseVocab):
  """"""
  _conll_idx = 10
  
  #=============================================================
  def __init__(self, token_vocab, *args, **kwargs):
    """"""
    
    super(TagRepVocab, self).__init__(*args, **kwargs)
    
    self._token_vocab = token_vocab
    self._matrix = []
    
    self.load(self.get('train_files')+'tag')
    self.load(self.get('parse_files')+'tag')
    return
  
  #=============================================================
  def __call__(self, placeholder=None, moving_params=None):
    """"""
    
    print('===tagrep_vocab.py: START Call ', ' name=', self.name, ' embed_size=', self._embed_size)
    embeddings = super(TagRepVocab, self).__call__(placeholder, moving_params=moving_params)
    
    #tf.summary.histogram('tagreph',embeddings)
    # (n x b x d') -> (n x b x d)
    with tf.variable_scope(self.name.title()):
      matrix = linalg.linear(embeddings, self._embed_size, moving_params=moving_params)
      if moving_params is None:
        with tf.variable_scope('Linear', reuse=True):
          weights = tf.get_variable('Weights')
          tf.losses.add_loss(tf.nn.l2_loss(tf.matmul(tf.transpose(weights), weights) - tf.eye(self._embed_size)))
    
    print('===tagrep_vocab.py: END Call ', ' name=', self.name)
    return matrix
    #return embeddings # changed in saves2/test8
  
  #=============================================================
  def load(self, filename):
    """"""
    
    embeddings = []
    cur_idx = 0
    

    open_func = codecs.open
    prefix = base64.urlsafe_b64encode(hashlib.sha1(os.path.abspath(filename)).digest()[0:10])[0:5]
    print("Ouverture de ",os.path.abspath(filename), " pour charger les embeddings\n")
    with open_func(filename, 'rb') as f:
        
      reader = codecs.getreader('utf-8')(f, errors='strict')
      
      for line_num, line in enumerate(reader):
          
        if len(line) > 3:
          tokenid = line.partition('\t')[0]
          line = line[line.index('\t'):len(line)]
          embeddings.append(np.fromstring(line, dtype=np.float32, sep=' '))
          
          self[prefix+'-'+tokenid] = cur_idx
          #print(prefix+'-'+tokenid)
          cur_idx += 1
    #try:
    embeddings = np.stack(embeddings)
    embeddings = np.pad(embeddings, ( (len(self.special_tokens),0), (0,0) ), 'constant')
    if self.matrix != []:
        self._matrix = np.concatenate((self._matrix,embeddings),axis=0)
    else:
        self._matrix = embeddings
    #self.embeddings = np.stack(embeddings)
    #self.embeddings = np.stack(self.matrix)
    #except:
      #shapes = set([embedding.shape for embedding in embeddings])
      #raise ValueError("Couldn't stack embeddings with shapes in %s" % shapes)
    return
  
  #=============================================================
 
  @property
  def conll_idx(self):
    return self._conll_idx
  @property
  def token_vocab(self):
    return self._token_vocab
  @property
  def token_embed_size(self):
    return (self.token_vocab or self).embed_size
  @property
  def matrix(self):
    return self._matrix
  @property
  def embeddings(self):
    return super(TagRepVocab, self).embeddings
  @embeddings.setter
  def embeddings(self, matrix):
    self._embed_size = matrix.shape[1]
    print(self._embed_size)
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.name.title()):
        self._embeddings = tf.Variable(matrix, name='Embeddings', trainable=False)
    return

#***************************************************************
if __name__ == '__main__':
  """"""
  
  tagrep_vocab = TagRepVocab(None)
  print('TagRepVocab passes')
