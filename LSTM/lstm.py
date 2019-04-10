#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/4/4
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
MAX_EPOCH = 5000
TIME_STEP = 28
INPUT_SIZE = 28
lr = 0.001
NUM_UNITS=50
N_CLASSES = 10

class build_lstm():
    def __init__(self):
        self.train_x = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP, INPUT_SIZE], name='x')
        self.train_y = tf.placeholder(tf.float32, name='y')

        with tf.variable_scope('lstm'):
            y_t, c_h = tf.nn.dynamic_rnn(
                # num_units即为lstm的output_size
                cell=tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS),
                inputs=self.train_x,
                initial_state=None,
                dtype=tf.float32,
                time_major=False
            )

        with tf.variable_scope('output_layer'):
            y = tf.layers.dense(inputs=y_t[:, -1, :], units=N_CLASSES)

        with tf.variable_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.train_y, logits=y)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    lstm = build_lstm()

    sess = tf.Session()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(MAX_EPOCH):
        for iteration in range(int(50000/BATCH_SIZE)):
            x, y = mnist.train.next_batch(BATCH_SIZE)
            x = np.reshape(x, [BATCH_SIZE, TIME_STEP, INPUT_SIZE])

            sess.run(lstm.train_op, feed_dict={lstm.train_x:x, lstm.train_y:y})
            loss = sess.run(lstm.loss, feed_dict={lstm.train_x:x, lstm.train_y:y})

            if iteration % 500 == 0:
                print("loss:", loss)