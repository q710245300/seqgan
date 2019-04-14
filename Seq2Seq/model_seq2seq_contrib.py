#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/4/10
import tensorflow as tf


class Seq2seq(object):

    def build_inputs(self, config):
        self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')

    def __init__(self, config, w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1):

        self.build_inputs(config)

        with tf.variable_scope("encoder"):
            # 初始化embedding层
            encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')
            # 相当于把输入经过embedding层
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)

            # 创建gru单元
            with tf.variable_scope("gru_cell"):
                encoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

            # 创建双向rnn层
            ((encoder_fw_outputs, encoder_bw_outputs),
             (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                                                 cell_bw=encoder_cell,
                                                                                                 inputs=encoder_inputs_embedded,
                                                                                                 sequence_length=self.seq_inputs_length,
                                                                                                 dtype=tf.float32,
                                                                                                 time_major=False)
            # 读取双向rnn层输出
            encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
            encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)

        with tf.variable_scope("decoder"):
            # decoder的embedding层
            decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='decoder_embedding')

            # 做一个size为[batch_size]的矩阵，用于存放，decoder输入的第一个‘_GO’
            tokens_go = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_GO') * w2i_target["_GO"]


            if useTeacherForcing:
                # 看use teacher forcing的图就能知道，因为预测的时候decoder的第一个输入是'_GO'，所以seq_targets[:, :-1]
                # reshape中的-1代表默认参数，自适应其他行，比如不关心或不知道无关紧要的行的大小，我就可以把改行设置为-1，让reshape自适应
                decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), self.seq_targets[:, :-1]], 1)
                helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs),
                                                           self.seq_targets_length)
            else:
                # 只要把‘_GO’,
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, w2i_target["_EOS"])

            with tf.variable_scope("gru_cell"):
                decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

                if useAttention:
                    if useBeamSearch > 1:
                        # 如果使用BeamSearch的话，输入到注意力机制的encoder的state和output的大小也都要改变，
                        # 而本代码的做法就是把encoder的输出直接复制k呗，存到注意力机制中
                        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=useBeamSearch)
                        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.seq_inputs_length,
                                                                              multiplier=useBeamSearch)
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim,
                                                                                   memory=tiled_encoder_outputs,
                                                                                   memory_sequence_length=tiled_sequence_length)
                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                        # 因为使用了 Beam Search，所以 decoder 的输入形状需要做 K 倍的扩展，tile_batch 就是用来干这个
                        tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                                                                  multiplier=useBeamSearch)
                        # 将 LSTM 中的状态初始化为全零，batch_size 给出一个 batch 的大小
                        tiled_decoder_initial_state = decoder_cell.zero_state(
                            batch_size=config.batch_size * useBeamSearch, dtype=tf.float32)
                        tiled_decoder_initial_state = tiled_decoder_initial_state.clone(
                            cell_state=tiled_encoder_final_state)
                        decoder_initial_state = tiled_decoder_initial_state
                    else:
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim,
                                                                                   memory=encoder_outputs,
                                                                                   memory_sequence_length=self.seq_inputs_length)
                        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.seq_inputs_length)
                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                        decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
                        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                else:
                    # 当不使用注意力机制的时候，就只要把encoder_state扩张大小就行了，不用将encoder_output大小也扩张
                    if useBeamSearch > 1:
                        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
                    else:
                        decoder_initial_state = encoder_state

            if useBeamSearch > 1:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, tokens_go,
                                                               w2i_target["_EOS"], decoder_initial_state,
                                                               beam_width=useBeamSearch,
                                                               output_layer=tf.layers.Dense(config.target_vocab_size))
            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                          output_layer=tf.layers.Dense(config.target_vocab_size))

            decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                       maximum_iterations=tf.reduce_max(
                                                                                                           self.seq_targets_length))

        if useBeamSearch > 1:
            self.out = decoder_outputs.predicted_ids[:, :, 0]
        else:
            decoder_logits = decoder_outputs.rnn_output
            self.out = tf.argmax(decoder_logits, 2)

            sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.seq_targets,
                                                         weights=sequence_mask)


            self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)