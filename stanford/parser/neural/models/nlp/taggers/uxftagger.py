#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2018 CEA Elie Duthoo (MODIFIED, original licence: Timothy Dozat)
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

import tensorflow as tf

from parser.neural.models.nlp.taggers.base_uxftagger import BaseUXFTagger

    
#***************************************************************
class UXFTagger(BaseUXFTagger):
  """"""
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """"""
    
    top_recur = super(UXFTagger, self).__call__(vocabs, moving_params=moving_params)
    int_tokens_to_keep = tf.to_int32(self.tokens_to_keep)
    
    featconds = [(vocab.startswith('feat') and vocab != 'feats') for vocab in self.vocabs]
    xtagconds = [(vocab.startswith('xtag') and vocab != 'xtags') for vocab in self.vocabs]
    nsplit = 2+sum(featconds)+sum(xtagconds)
    with tf.variable_scope('MLP'):
      mlps = (self.MLP(top_recur, self.mlp_size, n_splits=nsplit))
      tag_mlp, xtags_mlp, feats_mlp = mlps[0], mlps[1:1+sum(xtagconds)], mlps[1+sum(xtagconds)+1:]
    
    with tf.variable_scope('Tag'):
      tag_logits = self.linear(tag_mlp, len(self.vocabs['tags']))
      tag_probs = tf.nn.softmax(tag_logits)
      tag_preds = tf.to_int32(tf.argmax(tag_logits, axis=-1))
      tag_targets = self.vocabs['tags'].placeholder
      tag_correct = tf.to_int32(tf.equal(tag_preds, tag_targets))*int_tokens_to_keep
      tag_loss = tf.losses.sparse_softmax_cross_entropy(tag_targets, tag_logits, self.tokens_to_keep)
    
    xtag_correct_reduce = None
    xtag_logits,xtag_probs,xtag_preds,xtag_targets,xtag_correct,xtag_loss=list(),list(),list(),list(),list(),list()
    xtagvocabs = [i for (i, v) in zip(self.vocabs.keys(), xtagconds) if v]
    xtagvocabs = sorted(xtagvocabs)
    i=0
    for vocabname in xtagvocabs:
        with tf.variable_scope('Xtag'+vocabname[4:]):
          print("===uxftagger.py: ",vocabname[4:])
          vocab = self.vocabs[vocabname]
          xtag_logits.append(self.linear(xtags_mlp[i], len(vocab)))
          xtag_probs.append(tf.nn.softmax(xtag_logits[-1]))
          xtag_preds.append(tf.to_int32(tf.argmax(xtag_logits[-1], axis=-1)))
          xtag_targets.append(vocab.placeholder)
          xtag_correct.append(tf.to_int32(tf.equal(xtag_preds[-1], xtag_targets[-1]))*int_tokens_to_keep)
          xtag_loss.append(tf.losses.sparse_softmax_cross_entropy(xtag_targets[-1], xtag_logits[-1], self.tokens_to_keep))
          i+=1
          #tf.summary.histogram("xtag_correct_"+vocabname[4:],xtag_correct[-1])
          if xtag_correct_reduce == None:
              xtag_correct_reduce = xtag_correct[-1]
          else:
              xtag_correct_reduce = xtag_correct_reduce * xtag_correct[-1]
    
    feat_correct_reduce = None
    feat_logits,feat_probs,feat_preds,feat_targets,feat_correct,feat_loss=list(),list(),list(),list(),list(),list()
    featvocabs = [i for (i, v) in zip(self.vocabs.keys(), featconds) if v]
    featvocabs = sorted(featvocabs)
    i=0
    for vocabname in featvocabs:
        with tf.variable_scope('Feat'+vocabname[4:]):
          print("===uxftagger.py: ",vocabname[4:])
          vocab = self.vocabs[vocabname]
          feat_logits.append(self.linear(feats_mlp[i], len(vocab)))
          feat_probs.append(tf.nn.softmax(feat_logits[-1]))
          feat_preds.append(tf.to_int32(tf.argmax(feat_logits[-1], axis=-1)))
          feat_targets.append(vocab.placeholder)
          feat_correct.append(tf.to_int32(tf.equal(feat_preds[-1], feat_targets[-1]))*int_tokens_to_keep)
          feat_loss.append(tf.losses.sparse_softmax_cross_entropy(feat_targets[-1], feat_logits[-1], self.tokens_to_keep))
          i+=1
          #tf.summary.histogram("feat_correct_"+vocabname[4:],feat_correct[-1])
          if feat_correct_reduce == None:
              feat_correct_reduce = feat_correct[-1]
          else:
              feat_correct_reduce = feat_correct_reduce * feat_correct[-1]
    
    tagandfeat_correct = tag_correct * feat_correct_reduce
    correct = tag_correct * xtag_correct_reduce * feat_correct_reduce
    n_correct = tf.reduce_sum(correct)
    n_tag_correct = tf.reduce_sum(tag_correct)
    n_xtag_correct = tf.reduce_sum(xtag_correct_reduce)
    n_feat_correct = tf.reduce_sum(feat_correct_reduce)
    n_tagandfeat_correct = tf.reduce_sum(tagandfeat_correct)
    n_seqs_correct = tf.reduce_sum(tf.to_int32(tf.equal(tf.reduce_sum(correct, axis=1), self.sequence_lengths-1)))
    loss = tag_loss +  tf.reduce_sum(xtag_loss) + tf.reduce_sum(feat_loss)
     
    outputs = {
      'tag_logits': tag_logits,
      'tag_probs': tag_probs,
      'tag_preds': tag_preds,
      'tag_targets': tag_targets,
      'tag_correct': tag_correct,
      'tag_loss': tag_loss,
      'n_tag_correct': n_tag_correct,
      'tagandfeat_correct' : tagandfeat_correct,
      'n_tagandfeat_correct' : n_tagandfeat_correct,

      'xtag_logits': xtag_logits,
      'xtag_probs': xtag_probs,
      'xtag_preds': xtag_preds,
      'xtag_targets': xtag_targets,
      'xtag_correct': xtag_correct_reduce,
      'xtags_correct': xtag_correct,
      'xtag_loss': xtag_loss,
      'n_xtag_correct': n_xtag_correct,

      'feat_logits': feat_logits,
      'feat_probs': feat_probs,
      'feat_preds': feat_preds,
      'feat_targets': feat_targets,
      'feat_correct': feat_correct_reduce,
      'feats_correct': feat_correct,
      'feat_loss': feat_loss,
      'n_feat_correct': n_feat_correct,
      
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      'n_seqs_correct': n_seqs_correct,
      'loss': loss
      
      
    }
    
    
    return outputs
