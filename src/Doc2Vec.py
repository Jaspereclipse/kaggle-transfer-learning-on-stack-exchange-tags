#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function


import os.path as osp
import sys
import threading
import time

from data_factory import Feed
from Options import Options
import tensorflow as tf

class Doc2Vec(object):
    """
    Distributed Memory Model of Paragraph Vectors (PV-DM).
    """

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._feed = Feed(options.train_data, options.vocabulary, options.vocabulary_counts)
        self._options.vocab_size = self._feed.get_vocab_size()
        self._options.num_paragraphs = self._feed.get_num_paragraphs()
        self.build_graph()

    def forward(self, batch_words, batch_paragraphs, batch_labels):
        """forward passing of doc2vec net
        :param batch_words: words in the batch, e.g. [batch_size, window]
        :param batch_paragraphs: paragraphs in the batch, e.g. [batch_size, 1]
        :param batch_labels: labels in the batch, e.g. [batch_size, 1]
        :return: true label and contrast sample logits
        """

        opts = self._options

        # Word and paragraph embeddings
        wd_embed_init_width = 0.5 / opts.wd_emb_dim
        ph_embed_init_width = 0.5 / opts.ph_emb_dim
        wd_embed = tf.Variable(tf.random_uniform([opts.vocab_size, opts.wd_emb_dim], -wd_embed_init_width,
                                                 wd_embed_init_width), name="word_embedding")
        ph_embed = tf.Variable(tf.random_uniform([opts.num_paragraphs, opts.ph_emb_dim], -ph_embed_init_width,
                                                 ph_embed_init_width), name="paragraph_embedding")
        self._wd_embed = wd_embed
        self._ph_embed = ph_embed

        # Softmax weight & biases
        sm_wgt = tf.Variable(tf.zeros([opts.vocab_size, opts.input_emb_dim]), name="Softmax_weights")
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="Softmax_biases")

        # Global step
        self.global_step = tf.Variable(0, "global_step")

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
            unigrams=self._feed.get_vocal_counts()
        ))

        # Paragraph embeddings for the batch: [batch_size, ph_emb_dim]
        batch_ph_embed = tf.nn.embedding_lookup(ph_embed, batch_paragraphs)
        # Word embeddings for the batch: [batch_size, window, wd_emb_dim]
        batch_wd_embed = tf.nn.embedding_lookup(wd_embed, batch_words)
        # Input embeddings for the batch: [batch_size, input_emb_dim]
        if opts.mode == "sum":
            batch_input_embed = tf.add(tf.reduce_sum(batch_wd_embed, 1), batch_ph_embed)
        elif opts.mode == "average":
            batch_input_embed = tf.add(tf.reduce_mean(batch_wd_embed, 1), batch_ph_embed)
        else:
            batch_input_embed = tf.concat_v2([tf.reshape(batch_ph_embed, [opts.batch_size, opts.ph_emb_dim]),
                                              tf.reshape(batch_wd_embed, [opts.batch_size, opts.window * opts.wd_emb_dim])], 1)

        # Weights for labels: [batch_size, input_emb_dim]
        true_wgt = tf.nn.embedding_lookup(sm_wgt, batch_labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, batch_labels)

        # Weights for samples: [num_samples, input_emb_dim]
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
            targets=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            targets=tf.zeros_like(sampled_logits), logits=sampled_logits)
        loss = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / opts.batch_size
        return loss

    def optimize(self, loss):
        """sub-graph for optimize loss function
        :param loss: NCE loss
        :return: None
        """
        opts = self._options
        self._lr = tf.constant(opts.learning_rate, dtype=tf.float32, name='learning_rate')
        optimizer = tf.train.AdagradOptimizer(self._lr)
        train = optimizer.minimize(loss, global_step=self.global_step, gate_gradients=optimizer.GATE_NONE)
        self._train = train

    def build_graph(self):
        """graph for full doc2vec model"""
        opts = self._options

        batch_size = tf.constant(opts.batch_size, dtype=tf.int32)
        window = tf.constant(opts.window, dtype=tf.int32)
        batch_words, batch_paragraphs, labels, self._epoch = \
            tf.py_func(self._feed.get_batch, [batch_size, window], [tf.int32]*4)

        self._batch_words = batch_words
        self._batch_paragraphs = batch_paragraphs
        self._labels = labels

        print("Training data: %s"%opts.train_data)
        print("Vocabulary size (excluding UNK and NUL): %d"%(opts.vocab_size-2))
        print("Paragraphs per epoch: %d"%opts.num_paragraphs)

        true_logits, sampled_logits = self.forward(batch_words, batch_paragraphs, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar("NCE_loss", loss)
        self._loss = loss
        self.optimize(loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    # def _trainer(self): #
    #     """threads for training"""
    #     init_epoch, = self._session.run([self._epoch])
    #     while True:
    #         _, epoch = self._session.run([self._train, self._epoch])
    #         if epoch != init_epoch:
    #             break

    def train(self):
        """train doc2vec model for one epoch"""
        opts = self._options
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
        last_summary_time = 0
        last_checkpoint_time = 0

        init_epoch, = self._session.run([self._epoch])

        while True:
            _, epoch, step, loss = self._session.run([self._train, self._epoch, self.global_step, self._loss])
            now = time.time()
            if now - last_summary_time > opts.summary_interval:
                print("Epoch %4d Step %8d: loss = %6.2f" % (epoch, step, loss))
                sys.stdout.flush()
                summary = self._session.run(summary_op)
                summary_writer.add_summary(summary, step)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session, osp.join(opts.save_path, "model.ckpt"), global_step=step.astype(int))
                last_checkpoint_time = now
            if epoch != init_epoch:
                break

        return epoch

def main(_):
    """train doc2vec"""
    opts = Options()
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        with tf.device("/gpu:0"):
            model = Doc2Vec(opts, session)
        for _ in xrange(opts.epochs):
            model.train()
        model.saver.save(session, osp.join(opts.save_path, "model.ckpt"), global_step=model.global_step)

if __name__ == '__main__':
    tf.app.run()





