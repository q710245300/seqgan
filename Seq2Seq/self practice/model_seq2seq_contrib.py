import tensorflow as tf

class Seq2seq(object):
    def __init__(self, config, useTeacherForcing, useAttention, useBeamSearch):
        self.config = config

        self.seq_inputs = tf.placeholder(dtype=tf.float32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(dtype=tf.float32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(dtype=tf.float32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(dtype=tf.float32, name='seq_targets_length')

        with tf.variable_scope('Encoder'):
            # Embedding层相当于神经网络的参数，可以经过反向传播进行修正；因为是参数所以可以采用tf.Variable建立或者采用预训练的词向量
            encoder_embedding = tf.Variable(tf.random_uniform([config.input_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')

            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)

            with tf.variable_scope('gru_cell'):
                gru_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

            with tf.variable_scope('bidirectional_gru_layer'):
                '''
                time_major的意思就是time_step是不是第一列
                time_major: The shape format of the `inputs` and `outputs` Tensors.  
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.  
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                '''
                ((encoder_fw_outputs, encoder_bw_outputs),
                 (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell,
                                                                                                      cell_bw=gru_cell,
                                                                                                      inputs=encoder_inputs_embedded,
                                                                                                      sequence_length=self.seq_inputs_length,
                                                                                                      dtype=tf.float32,
                                                                                                      time_major=False)

                # 将双向rnn的两个rnn fw和bw的输出合并
                encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)
                encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)

        '''
        decoder的输入是’_GO‘加删去最后一个字的target
        '''
        with tf.variable_scope('decoder'):
            decoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='decoder_embedding')

            tokens_go = tf.ones([config.batch_size, 1], dtype=tf.int32, name='tokens_go') * config.w2i_target["_GO"]

            if useTeacherForcing:
                decoder_inputs = tf.concat([tokens_go, self.seq_targets], name='decoder_inputs')
                decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)

                #
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, self.seq_targets_length)
            else:
                helper = tf.contirb.seq2seq.GreedyEmbeddingHelper(decoder_embedding, config.w2i_target["_GO"], config.w2i_target["_EOS"])

            with tf.variable_scope('decoder_cell'):
                decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

                if useAttention:
                    if useBeamSearch > 1:
                        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=useBeamSearch)
                        tiled_sequences_length = tf.contrib.seq2seq.tile_batch(self.seq_inputs_length, multiplier=useBeamSearch)

                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim,
                                                                                   memory=tiled_encoder_outputs,
                                                                                   memory_sequence_length=tiled_sequences_length)

                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)

                        tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multipiler=useBeamSearch)
                        decoder_initial_state = decoder_cell.zero_state(
                            batch_size=config.batch_size*useBeamSearch, dtype=tf.float32
                        ).clone(cell_state=tiled_encoder_final_state)

                    else:
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim,
                                                                                   memory = encoder_outputs,
                                                                                   memory_sequence_length=self.seq_inputs_length)

                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)

                        decoder_initial_state = decoder_cell.zero_state(
                            batch_size=config.batch_size,dtype=tf.float32
                        ).clone(cell_state=encoder_state)

                else:
                    if useBeamSearch > 1:
                        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
                    else:
                        decoder_initial_state = encoder_state

            if useBeamSearch > 1:
                 decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, config.w2i_target["_GO"],
                                                                config.w2i_target["_EOS"], decoder_initial_state,
                                                                beam_width=useBeamSearch,
                                                                output_layer=tf.layers.Dense(config.target_vocab_size))
            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                          output_layer=tf.layers.Dense(config.target_vocab_size))

            decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                       maximun_iterations=tf.reduce_max(
                                                                                                           self.seq_targets_length))
