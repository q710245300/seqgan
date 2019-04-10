#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/4/5
# c = np.random.random([10, 1])  # 随机生成一个10*1的数组
# b = tf.nn.embedding_lookup(c, [1, 3])#查找数组中的序号为1和3的
import tensorflow as tf

p = tf.Variable(tf.random_normal([10, 1]))  # 生成10*1的张量
b = tf.nn.embedding_lookup(p, [1, 3])  # 查找张量中的序号为1和3的

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('embeded p', sess.run(p))
    print('embeded b',sess.run(b))