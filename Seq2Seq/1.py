#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/4/5
import tensorflow as tf


class Seq2seq(object):
    def __init__(self, config, w2i_target):
        self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')

        with tf.variable_scope("encoder"):
            encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)
            encoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs_embedded,
                                                               sequence_length=self.seq_inputs_length, dtype=tf.float32,
                                                               time_major=False)

        tokens_go = tf.ones([config.batch_size], dtype=tf.int32) * w2i_target["_GO"]
        # [:-1]除了最后一个元素的其余所有元素，（可能target最后一个元素放的是终止符_EOS,所以剔除）
        # 经过tf.concat的拼接，相当于拼接成大小和原始target相同，第一个元素为_GO, 剩下元素为除了终止符的回答
        # 例如开始target为‘你瞅啥_EOS’变为‘_GO你瞅啥’
        decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), self.seq_targets[:, :-1]], 1)

        with tf.variable_scope("decoder"):
            decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='decoder_embedding')
            decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
            decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_inputs_embedded,
                                                               initial_state=encoder_state,
                                                               sequence_length=self.seq_targets_length,
                                                               dtype=tf.float32, time_major=False)

        # 输出大小为[batch_size, time_steps, config.target_vocab_size]
        # 每个batch输出time_steps个词，每个词都从目标词汇表中选取， 最大的值就是要输出的词,最后time_stpes个词构成一句话
        decoder_logits = tf.layers.dense(decoder_outputs.rnn_output, config.target_vocab_size)
        self.out = tf.argmax(decoder_logits, 2)