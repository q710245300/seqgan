import tensorflow as tf

class Seq2seq(object):
    def __init__(self, config):
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
                ((encoder_fw_outputs, encoder_bw_outputs),
                 (encoder_fw_fianal_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell,
                                                                                                      cell_bw=gru_cell,
                                                                                                      inputs=encoder_inputs_embedded,
                                                                                                      sequence_length=self.seq_inputs_length,
                                                                                                      dtype=tf.float32,
                                                                                                      time_major=False)