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

import os
import time
import codecs
import cPickle as pkl
from collections import defaultdict
import os.path as op
import sys
import numpy as np
import tensorflow as tf

from parser import Configurable
from parser.vocabs import *
from parser.dataset import *
from parser.misc.colors import ctext
from parser.neural.optimizers import RadamOptimizer
from tensorflow.python.tools import freeze_graph

#***************************************************************
class Network(Configurable):
  """"""

  #=============================================================
  def __init__(self, train=False,*args, **kwargs):
    """"""

    super(Network, self).__init__(*args, **kwargs)
    # hacky!
    #hacky_train_files = op.join(self.save_dir, op.basename(self.get("train_files")))
    #self._config.set('Configurable', 'train_files', hacky_train_files)
    
    #todo: essayer de reserver tt le gpu avant toute chose, si ça n'est pas possible on passe au prochain gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    temp_nlp_model = self.nlp_model.from_configurable(self)

    print ("\nloading word vocab")
    word_vocab = WordVocab.from_configurable(self)
    
    print ("\nloading pretrained vocab")
    pretrained_vocab = PretrainedVocab.from_vocab(word_vocab)
    
    print ("\nloading tag representation vocab")
    tagrep_vocab = None
    if 'tagrep' in temp_nlp_model.input_vocabs:
        tagrep_vocab = TagRepVocab.from_vocab(word_vocab)
    
    print ("\nloading subtoken vocab")
    subtoken_vocab = self.subtoken_vocab.from_vocab(word_vocab)
    
    print ("\ncreating multivocab")
    vocabs_of_multivocab = [word_vocab, pretrained_vocab]
    word_multivocab = Multivocab.from_configurable(self, vocabs_of_multivocab, name=word_vocab.name)
    if ('chars' in temp_nlp_model.input_vocabs or word_multivocab.merge_strategy != "only_pretrained") :
        vocabs_of_multivocab.append(subtoken_vocab)
        word_multivocab = Multivocab.from_configurable(self, vocabs_of_multivocab, name=word_vocab.name)
    print(word_multivocab.merge_strategy)
    print ('chars' in temp_nlp_model.input_vocabs );
    print (word_multivocab.merge_strategy != "only_pretrained" );
 
    print ("\nloading tag vocab")
    tag_vocab = None
    if 'tags' in temp_nlp_model.input_vocabs or 'tags' in temp_nlp_model.output_vocabs:
        tag_vocab = TagVocab.from_configurable(self)
    
    print ("\nloading dep vocab")
    dep_vocab = DepVocab.from_configurable(self)
    
    print ("\nloading lemma vocab")
    lemma_vocab = LemmaVocab.from_configurable(self)
    
    print ("\nloading xtag vocab")
    xtag_vocab = None
    if 'xtags' in temp_nlp_model.input_vocabs or 'xtags' in temp_nlp_model.output_vocabs or 'subxtags' in temp_nlp_model.input_vocabs or 'subxtags' in temp_nlp_model.output_vocabs:
        xtag_vocab = XTagVocab.from_configurable(self)
    
    print ("\nloading feat vocab")
    feat_vocab = None
    if 'feats' in temp_nlp_model.input_vocabs or 'feats' in temp_nlp_model.output_vocabs or 'subfeats' in temp_nlp_model.input_vocabs or 'subfeats' in temp_nlp_model.output_vocabs:
        feat_vocab = FeatVocab.from_configurable(self)
    
    print ("\nloading subfeat vocab")
    subfeat_vocabs = None
    if 'subfeats' in temp_nlp_model.input_vocabs or 'subfeats' in temp_nlp_model.output_vocabs:
        subfeat_vocabs = self.prepare_subpos_vocabs(feat_vocab,temp_nlp_model);
    print ("\nloading subxtag vocab")
    subxtag_vocabs = None
    if 'subxtags' in temp_nlp_model.input_vocabs or 'subxtags' in temp_nlp_model.output_vocabs:
        subxtag_vocabs = self.prepare_subpos_vocabs(xtag_vocab,temp_nlp_model);
    
    
    head_vocab = HeadVocab.from_configurable(self)
    rel_vocab = RelVocab.from_configurable(self)
    self._vocabs = [dep_vocab, word_multivocab, lemma_vocab, tag_vocab, xtag_vocab, feat_vocab, head_vocab, rel_vocab, tagrep_vocab]
    if subfeat_vocabs != None:
        self._vocabs.extend(subfeat_vocabs)
    if subxtag_vocabs != None:
        self._vocabs.extend(subxtag_vocabs)
    self._vocabs = filter(None, self._vocabs)
    print(self._vocabs)
    #print('subfeat_vocabs:',len(subfeat_vocabs))
    #print('subxtag_vocabs:',len(subxtag_vocabs))
    
    self._global_step = tf.Variable(0., trainable=False, name='global_step')
    self._global_epoch = tf.Variable(0., trainable=False, name='global_epoch')
    self._optimizer = RadamOptimizer.from_configurable(self, global_step=self.global_step)
    if train:
        self._global_time = tf.Variable(0., trainable=False, name='global_time')
    return

  #=============================================================
  def prepare_subpos_vocabs(self, ori_vocab,temp_nlp_model):
    tb_xtag_sep = dict()
    tb_xtag_sep['ko_kaist'] = '+'
    tb_xtag_sep['ko_gsd'] = '+'
    tb_xtag_sep['gl_ctg'] = None
    tb_xtag_sep['vi_vtb'] = None
    tb_xtag_sep['zh_gsd'] = None
    tb_xtag_sep['nl_lassysmall'] = '|'
    tb_xtag_sep['nl_alpino'] = '|'
    tb_xtag_sep['af_afribooms'] = None
    tb_xtag_sep['en_lines'] = '-'
    tb_xtag_sep['la_ittb'] = '|'
    tb_xtag_sep['th_forged'] = '|'
    tb_xtag_sep['th_pud'] = '|'
    
    print("train:"+str(self.train_files))
    traindata = self.train_files[0]
    tb = self.lc
    
    
    if ori_vocab.name == 'feats':
        subfeat_vocabs = list();
        ori_vocab.separator = '|'
            
        if temp_nlp_model.__class__.__name__ == 'Parser' and tb in tb_xtag_sep:
            subfeat_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'feat'))
        else:
            if not os.path.exists( self.save_dir+"/featlist.cfg"):
                os.system("cat "+self.train_files[0]+"| grep -v '^#'|grep -v '^=' |sed '/^\s*$/d'|cut -f6 \
                                       |sort -u |tr '|' '\n' |sort -u |sed s/=.*//g |sort -u > "+ self.save_dir+"/featlist.cfg")
            #this was only made to tackle the read-only server on tira
            f = os.popen("cat "+self.train_files[0]+"| grep -v '^#'|grep -v '^=' |sed '/^\s*$/d'|cut -f6 \
                                       |sort -u |tr '|' '\n' |sort -u |sed s/=.*//g |sort -u")
            categories = f.read()
            categories = categories.split('\n')
            if '_' in categories: categories.remove('_')
            if '' in categories: categories.remove('')
            for category in categories:
                category = category.replace('[','')
                category = category.replace(']','')
                subfeat_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'feat'+category))
                
            if not subfeat_vocabs:
                subfeat_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'feat_'))
                
        #subfeat_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'featall'))
        return subfeat_vocabs
    
    if ori_vocab.name == 'xtags':
        subxtag_vocabs = list();
        if tb in tb_xtag_sep:
            ori_vocab.separator = tb_xtag_sep[tb]
        else:
            ori_vocab.separator = None
        if tb not in tb_xtag_sep: #same as no xtags
            subxtag_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'xtag'))
        else:
            if tb_xtag_sep[tb] != None:
                if not os.path.exists( self.save_dir+"/xtaglist.cfg"):
                    f = os.popen("cat "+traindata+"| grep -v '^#'|grep -v '^=' |sed '/^\s*$/d'|cut -f5|sort -u \
                                 |tr '"+tb_xtag_sep[tb]+"' '\n' |sort -u > "+ self.save_dir+"/xtaglist.cfg")
                f = os.popen("cat "+traindata+"| grep -v '^#'|grep -v '^=' |sed '/^\s*$/d'|cut -f5|sort -u \
                                 |tr '"+tb_xtag_sep[tb]+"' '\n' |sort -u")
                categories = f.read()
                categories = categories.split('\n')
                if '_' in categories: categories.remove('_')
                if '' in categories: categories.remove('')
                for category in categories:
                    category = category.replace('[','')
                    category = category.replace(']','')
                    try:
                        vocab = self.subpos_vocab.from_vocab(ori_vocab,'xtag'+category)
                        subxtag_vocabs.append(vocab)
                    except ValueError:
                        print(category,' : incorrect name for tensorflow (skipped)')
            else:
                if not os.path.exists( self.save_dir+"/xtaglist.cfg"):
                    f = os.popen("cat "+traindata+"| grep -v '^#'|grep -v '^=' |sed '/^\s*$/d'|cut -f5|sort -u > "+ self.save_dir+"/xtaglist.cfg")
                f = os.popen("cat "+traindata+"| grep -v '^#'|grep -v '^=' |sed '/^\s*$/d'|cut -f5|sort -u")
                categories = f.read()
                categories = categories.split('\n')
                if '_' in categories: categories.remove('_')
                if '' in categories: categories.remove('')
                subxtag_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'xtagall'))
                
        if not subxtag_vocabs:
            subxtag_vocabs.append(self.subpos_vocab.from_vocab(ori_vocab,'xtag_'))
                
        return subxtag_vocabs
        
    return None

  #=============================================================
  def add_file_vocabs(self, conll_files):
    """"""

    # TODO don't depend on hasattr
    for vocab in self.vocabs:
      if hasattr(vocab, 'add_files'):
        vocab.add_files(conll_files)
    for vocab in self.vocabs:
      if hasattr(vocab, 'index_tokens'):
        vocab.index_tokens()
      if vocab.name == 'tagrep':
        for conll_file in conll_files:
          vocab.load(conll_file+'tag')
        vocab.embeddings = np.stack(vocab.matrix)
        vocab._matrix = None;
    return

  #=============================================================
  def train(self, load=False):
    """"""
    print('TRAIN')
    
    # prep the configurables
    self.add_file_vocabs(self.parse_files)

    #train
    trainset = Trainset.from_configurable(self, self.vocabs, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title()):
      train_tensors = trainset() 
    train = self.optimizer(tf.losses.get_total_loss())
    train_outputs = [train_tensors[train_key] for train_key in trainset.train_keys]
    saver = tf.train.Saver(self.save_vars, max_to_keep=1)

    #valid
    validset = Parseset.from_configurable(self, self.vocabs, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title(), reuse=True):
      valid_tensors = validset(moving_params=self.optimizer)
    valid_outputs = [valid_tensors[train_key] for train_key in validset.train_keys]
    valid_outputs2 = [valid_tensors[valid_key] for valid_key in validset.valid_keys]

    #init
    current_acc = 0
    best_acc = 0
    n_iters_since_improvement = 0
    n_iters_in_epoch = 0
    # calling these properties is inefficient so we save them in separate variables
    min_train_iters = self.min_train_iters
    max_train_time = self.max_train_time
    min_percent_per_hour = self.min_percent_per_hour
    validate_every = self.validate_every
    save_every = self.save_every
    verbose = self.verbose
    quit_after_n_iters_without_improvement = self.quit_after_n_iters_without_improvement
    


    #if no load file, don't load
    if not op.isfile(os.path.join(self.save_dir, 'history.pkl')):
        load = False;
    # load or prep the history
    if load :
      self.history = pkl.load(open(os.path.join(self.save_dir, 'history.pkl')))
    else :
      self.history = {'train': defaultdict(list), 'valid': defaultdict(list)}


    #stoping criteria
    range_scores = 20;
    last_scores = np.repeat(0.0,range_scores);
    last_train_time = np.repeat(1.0,range_scores);

    # start up the session
    last_training_time = time.time()

    #configuration du processeur
    config_proto = tf.ConfigProto();#allow_soft_placement=True) #log_device_placement=True,allow_soft_placement=Tru

    #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.5
    #config_proto.gpu_options.allow_growth = True
    
    with tf.Session(config=config_proto) as sess: #initialisation de la session
      #train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)

      sess.run(tf.global_variables_initializer())
      if load: #Restaure une session
        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
      total_train_iters = sess.run(self.global_step)
      train_accumulators = np.zeros(len(train_outputs))
      train_time = 0
      # training loop
      while sess.run(self.global_time) < max_train_time:
        for feed_dict in trainset.iterbatches():
          sys.stdout.write("-")
          sys.stdout.flush()

          start_time = time.time()
          #print(sess.run(train_outputs + [train], feed_dict=feed_dict)[:-1])

          batch_values = sess.run(train_outputs + [train], feed_dict=feed_dict)[:-1]
          batch_time = time.time() - start_time
          
          #merge = tf.summary.merge_all()
          #summary = sess.run(merge, feed_dict=feed_dict)
          #train_writer.add_summary(summary, total_train_iters)


          # update accumulators
          total_train_iters += 1
          n_iters_since_improvement += 1
          train_accumulators += batch_values
          train_time += batch_time
          

          sess.run(self.global_time.assign_add((time.time() - last_training_time)/60.0))
          last_training_time = time.time()

          # possibly validate
          if total_train_iters == 1 or (total_train_iters % validate_every == 0):
            valid_accumulators = np.zeros(len(train_outputs))
            valid_time = 0
            print(ctext('\nStarting sanity check...', 'bright_yellow'))
            with codecs.open(os.path.join(self.save_dir, 'sanity_check'), 'w', encoding='utf-8', errors='ignore') as f:
              for feed_dict, sents in validset.iterbatches(return_check=True):
                start_time = time.time()
                batch_values = sess.run(valid_outputs+valid_outputs2, feed_dict=feed_dict)
                batch_time = time.time() - start_time
                # update accumulators
                valid_accumulators += batch_values[:len(valid_outputs)]
                valid_preds = batch_values[len(valid_outputs):]
                valid_time += batch_time
                
                #validset.check(valid_preds, sents, f)
            print(ctext('End of sanity check', 'bright_yellow'))
            # update history
            trainset.update_history(self.history['train'], train_accumulators)
            current_acc = validset.update_history(self.history['valid'], valid_accumulators)
            # print
            if verbose:
              print(ctext('{0:6d}'.format(int(total_train_iters)), 'bold')+')')
              trainset.print_accuracy(train_accumulators, train_time)
              validset.print_accuracy(valid_accumulators, valid_time)
              print("Train since: ",sess.run(self.global_time),"min")
            train_accumulators = np.zeros(len(train_outputs))
            
            
            #evaluation du critere d'arret
            last_scores[range_scores-1] = current_acc
            last_train_time[range_scores-1] = train_time
            
            avg_percent_per_sec = np.diff(last_scores) / last_train_time[1:]
            avg_percent_per_sec = np.mean(avg_percent_per_sec)
            avg_percent_per_hour = avg_percent_per_sec * 3600
            print((int)(avg_percent_per_hour),"% par heure")
            
            stop_criteria = avg_percent_per_hour < min_percent_per_hour and total_train_iters > min_train_iters;
            stop_criteria = stop_criteria or sess.run(self.global_time) > max_train_time;
            #logs
            print("avg_percent_per_hour ",avg_percent_per_hour)
            print("min_percent_per_hour ",min_percent_per_hour)
            print("total_train_iters ",total_train_iters)
            print("min_train_iters ",min_train_iters)
            print("global_time ",sess.run(self.global_time))
            print("max_train_time ",max_train_time)
            print("avg_percent_per_hour < min_percent_per_hour ",avg_percent_per_hour < min_percent_per_hour)
            print("total_train_iters > min_train_iters ",total_train_iters > min_train_iters)
            print("sess.run(self.global_time) > max_train_time ",sess.run(self.global_time) > max_train_time)
            print("stop_criteria ",stop_criteria)
            print("best acc ",best_acc)
            
            #Best model
            if current_acc > best_acc:
              if verbose:
                print(ctext('Saving model...', 'bright_yellow'))
              best_acc = current_acc
              n_iters_since_improvement = 0
              saver.save(sess, os.path.join(self.save_dir, self.name.lower()),
                         #global_step=self.global_epoch,
                         write_meta_graph=False)
              with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
                pkl.dump(dict(self.history), f)
              if verbose:
                print(ctext('Saved!', 'bright_yellow'))
            #Stopping criteria
            if stop_criteria:
              break
            
           #shift stopping criteria arrays
            for i in range(1,range_scores):
                last_scores[i-1] = last_scores[i]
                last_train_time[i-1] = last_train_time[i]
            
            train_time = 0
        else:
          # We've completed one epoch
          if total_train_iters <= min_train_iters:
            saver.save(sess, os.path.join(self.save_dir, self.name.lower()),
                       #global_step=self.global_epoch,
                       write_meta_graph=False)
            with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
              pkl.dump(dict(self.history), f)
          sess.run(self.global_epoch.assign_add(1.))
          continue
        break
      # Now parse the training and testing files
      input_files = self.train_files + self.parse_files
      saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
      for input_file in input_files:
        parseset = Parseset.from_configurable(self, self.vocabs, parse_files=input_file, nlp_model=self.nlp_model)
        with tf.variable_scope(self.name.title(), reuse=True):
          parse_tensors = parseset(moving_params=self.optimizer)
        parse_outputs = [parse_tensors[parse_key] for parse_key in parseset.parse_keys]

        input_dir, input_file = os.path.split(input_file)
        output_dir = self.save_dir
        output_file = input_file

        start_time = time.time()
        probs = []
        sents = []
        for feed_dict, tokens in parseset.iterbatches(shuffle=False):
          probs.append(sess.run(parse_outputs, feed_dict=feed_dict))
          sents.append(tokens)
        parseset.write_probs(sents, os.path.join(output_dir, output_file), probs)
    if self.verbose:
      print(ctext('Parsing {0} file(s) took {1} seconds'.format(len(input_files), time.time()-start_time), 'bright_green'))
    return

  #=============================================================
  def parse(self, input_files, output_dir=None, output_file=None):
    """"""
    print('PARSE')

    if not isinstance(input_files, (tuple, list)):
      input_files = [input_files]
    if len(input_files) > 1 and output_file is not None:
      raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
    self.add_file_vocabs(input_files)

    # load the model and prep the parse set
    trainset = Trainset.from_configurable(self, self.vocabs, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title()):
      train_tensors = trainset()
    train_outputs = [train_tensors[train_key] for train_key in trainset.train_keys]

    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    #saver = tf.train.Saver()
    config_proto = tf.ConfigProto()
    if self.per_process_gpu_memory_fraction == -1:
      config_proto.gpu_options.allow_growth = True
    else:
      config_proto.gpu_options.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
    with tf.Session(config=config_proto) as sess:
      #import pdb; pdb.set_trace()
      for var in self.non_save_vars:
        sess.run(var.initializer)

      saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

      # Iterate through files and batches
      for input_file in input_files:
        parseset = Parseset.from_configurable(trainset, self.vocabs, parse_files=input_file, nlp_model=self.nlp_model)
        with tf.variable_scope(self.name.title(), reuse=True):
          parse_tensors = parseset(moving_params=self.optimizer)
        parse_outputs = [parse_tensors[parse_key] for parse_key in parseset.parse_keys]
        parse_outputs_valid = [parse_tensors[tk] for tk in parseset.train_keys]
        parse_outputs_valid2 = [parse_tensors[vk] for vk in parseset.valid_keys]

        input_dir, input_file = os.path.split(input_file)
        if output_dir is None and output_file is None:
          output_dir = self.save_dir
        if output_dir == input_dir and output_file is None:
          output_path = os.path.join(input_dir, 'parsed-'+input_file)
        elif output_file is None:
          output_path = os.path.join(output_dir, input_file)
        else:
          output_path = output_file

        start_time = time.time()
        probs = []
        sents = []
        arc_corr, n_tokens,corr = 0.,0.,0.
        
        #with codecs.open(os.path.join(self.save_dir, 'sanity_check_parse'), 'w', encoding='utf-8', errors='ignore') as f:
        #  for feed_dict, sentss in parseset.iterbatches(return_check=True):
        #    batch_values = sess.run(parse_outputs_valid+parse_outputs_valid2, feed_dict=feed_dict)
        #    valid_preds = batch_values[len(parse_outputs_valid):]
        #    parseset.check(valid_preds, sentss, f)
            
        #Ici tokens ne contient déjà plus les "du = de le"
        for feed_dict, tokens in parseset.iterbatches(shuffle=False):
          results = sess.run(parse_outputs+train_outputs, feed_dict=feed_dict)
          probs.append(results[:3])
          sents.append(tokens)
          arc_corr+=results[7]
          n_tokens+=results[3]
          corr+=results[8]
        UAS = arc_corr/n_tokens*100
        LAS = corr/n_tokens*100
        metrics={'UAS':UAS,'LAS':LAS}
        parseset.write_probs(sents, output_path, probs)
        #tf.summary.FileWriter(self.save_dir,sess.graph)


    #self.__freeze_my_graph__(sess)
    if self.verbose:
      print('Parsing {0} file(s) took {1} seconds'.format(len(input_files), time.time()-start_time))
      msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
      print(msg)
    return



  #=============================================================
  def tag(self, input_files, output_dir=None, output_file=None):
    """"""

    if not isinstance(input_files, (tuple, list)):
      input_files = [input_files]
    if len(input_files) > 1 and output_file is not None:
      raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
    self.add_file_vocabs(input_files)

    # load the model and prep the parse set
    trainset = Trainset.from_configurable(self, self.vocabs, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title()):
      train_tensors = trainset()
    train_outputs = [train_tensors[train_key] for train_key in trainset.train_keys]

    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    #saver = tf.train.Saver()
    config_proto = tf.ConfigProto()
    if self.per_process_gpu_memory_fraction == -1:
      config_proto.gpu_options.allow_growth = True
    else:
      config_proto.gpu_options.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
    with tf.Session(config=config_proto) as sess:
      #import pdb; pdb.set_trace()
      for var in self.non_save_vars:
        sess.run(var.initializer)

      saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

      # Iterate through files and batches
      for input_file in input_files:
        parseset = Parseset.from_configurable(trainset, self.vocabs, parse_files=input_file, nlp_model=self.nlp_model)
        with tf.variable_scope(self.name.title(), reuse=True):
          parse_tensors = parseset(moving_params=self.optimizer)
        parse_outputs = [parse_tensors[parse_key] for parse_key in parseset.parse_keys]

        input_dir, input_file = os.path.split(input_file)
        if output_dir is None and output_file is None:
          output_dir = self.save_dir
        if output_dir == input_dir and output_file is None:
          output_path = os.path.join(input_dir, 'tagged-'+input_file)
        elif output_file is None:
          output_path = os.path.join(output_dir, input_file)
        else:
          output_path = output_file

        start_time = time.time()
        probs = []
        sents = []
        
        for feed_dict, tokens in parseset.iterbatches(shuffle=False):
          results = sess.run(parse_outputs+train_outputs, feed_dict=feed_dict)
          probs.append(results[:len(parseset.parse_keys)])
          sents.append(tokens)
        parseset.write_probs(sents, output_path, probs)
        #tf.summary.FileWriter(self.save_dir,sess.graph)


    #self.__freeze_my_graph__(sess)
    return

  #=============================================================
  @property
  def vocabs(self):
    return self._vocabs
  @property
  def datasets(self):
    return self._datasets
  @property
  def optimizer(self):
    return self._optimizer
  @property
  def save_vars(self):
    return filter(lambda x: not x.name in [u'Pretrained/Embeddings:0',u'Tagrep/Embeddings:0'], tf.global_variables())
  @property
  def non_save_vars(self):
    return filter(lambda x: x.name in [u'Pretrained/Embeddings:0',u'Tagrep/Embeddings:0'] , tf.global_variables())
  @property
  def global_step(self):
    return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def global_time(self):
    return self._global_time

#***************************************************************
if __name__ == '__main__':
  """"""

  from parser import Network
  configurable = Configurable()
  network = Network.from_configurable(configurable)
  network.train()
