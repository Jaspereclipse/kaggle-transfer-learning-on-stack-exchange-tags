#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import threading
import time

import numpy as np
import tensorflow as tf

class Doc2Vec(object):
    """
    Distributed Memory Model of Paragraph Vectors (PV-DM).
    """

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._wd_embed = None
        self._ph_embed = None
        self.global_step = None

    def forward(self, batch_data, batch_labels):
        """forward passing of doc2vec net
        :param batch_data: "paragraph": [batch_size, 1]; "word": [batch_size, window]
        :param batch_labels: [batch_size, 1]
        :return: true label and contrast sample logits
        """

        opts = self._options

        # Word and paragraph embeddings
        wd_embed_init_width = 0.5 / opts.wd_emb_dim
        ph_embed_init_width = 0.5 / opts.ph_emb_dim
        wd_embed = tf.Variable(tf.random_uniform([opts.vocab_size, opts.wd_embed_dim], -wd_embed_init_width,
                                                 wd_embed_init_width), name="word_embedding")
        ph_embed = tf.Variable(tf.random_uniform([opts.num_paragraphs, opts.ph_embed_din], -ph_embed_init_width,
                                                 ph_embed_init_width), name="paragraph_embedding")
        self._wd_embed = wd_embed
        self._ph_embed = ph_embed

        # Softmax weight & biases
        sm_wgt = tf.Variable(tf.zeros([opts.vocab_size, opts.input_embed_dim]), name="Softmax weights")
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="Softmax biases")

        # Global step
        self.global_step = tf.Variable(0, "global step")

        # Prepare for computing NCE loss
        labels_matrix = tf.reshape(tf.cast(batch_labels, dtype=tf.int64), [opts.batch_size, 1])

        # Negative sampling
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=.75,
            unigrams=opts.vocab_counts.tolist()
        ))

        # Paragraph embeddings for the batch: [batch_size, ph_embed_dim]
        batch_ph_embed = tf.nn.embedding_lookup(ph_embed, batch_data["paragraph"])

        # Word embeddings for the batch: [batch_size, window, wd_embed_dim]
        batch_wd_embed = tf.nn.embedding_lookup(wd_embed, batch_data["word"])

        # Input embeddings for the batch: [batch_size, input_embed_dim]
        if opts.mode == "sum":
            batch_input_embed = tf.add(tf.reduce_sum(batch_wd_embed, 1), batch_ph_embed)
        elif opts.mode == "average":
            batch_input_embed = tf.add(tf.reduce_mean(batch_wd_embed, 1), batch_ph_embed)
        else:
            batch_input_embed = tf.concat(1, [batch_ph_embed, tf.reshape(batch_wd_embed, [opts.batch_size, opts.window * opts.wd_emb_dim])])

        # Weights for labels: [batch_size, input_embed_dim]
        true_wgt = tf.nn.embedding_lookup(sm_wgt, batch_labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, batch_labels)

        # Weights for samples: [num_samples, input_embed_dim]
        sampled_wgt = tf.nn.embedding_lookup(sm_wgt, sampled_ids)
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(batch_input_embed, true_wgt), 1) + true_b

        # Sampled logits: [batch_size, num_samples]
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(batch_input_embed, sampled_wgt, transpose_b=True) + sampled_b_vec

        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """sub-graph for computing NCE loss
        :param true_logits:
        :param sampled_logits:
        :return: NCE loss
        """
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        loss = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_logits)) / opts.batch_size
        return loss

    def optimize(self, loss):
        """sub-graph for optimize loss function
        :param loss: NCE loss
        :return: None
        """
        opts = self._options
        pass