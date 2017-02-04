#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from _init_ import config
from data_factory import load_pickle
import os
import sys
import threading
import time

import numpy as np
import tensorflow as tf

# takes memory
word2id = load_pickle(config['vocabulary'])
id2word = {v: k for k, v in word2id.iteritems()}
df_sent = load_pickle(config['training_data'])

flags = tf.app.flags
flags.DEFINE_string("train_data", None, "Doc2Vec training data")
flags.DEFINE_integer("doc_embedding_size", 200, "embedding size for paragraph")
flags.DEFINE_integer("word_embedding_size", 100, "embedding size for word")
flags.DEFINE_integer("vocab_size", len(word2id), "vocabulary size")
flags.DEFINE_integer("num_paragraphs", len(set(df_sent['qid'])), "number of paragraphs in the training set")
flags.DEFINE_boolean("concat", True, "whether to concatenate paragraph embedding with word embeddings")
flags.DEFINE_boolean("sum_", False, "whether to sum all embeddings (both paragraph and word)")
flags.DEFINE_boolean("average", False, "whether to average all embeddings (both paragraph and word)")
flags.DEFINE_integer("window", 4, "number of preceding words used to predict next word")

FLAGS = flags.FLAGS