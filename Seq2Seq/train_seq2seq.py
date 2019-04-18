#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/4/10
import tensorflow as tf
import numpy as np
import random
import time
from model_seq2seq_contrib import Seq2seq

# from model_seq2seq import Seq2seq

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True


class Config(object):
    def __init__(self, w2i_source, w2i_target):
        self.embedding_dim = 100
        self.hidden_dim = 50
        self.batch_size = 128
        self.learning_rate = 0.005
        self.w2i_source = w2i_source
        self.w2i_target = w2i_target
        self.source_vocab_size = len(self.w2i_source)
        self.target_vocab_size = len(self.w2i_target)

# 构建样本集train_set
# 返回两个大小为10000, 内容分别为123，数字的英文翻译的数据集
def load_data(path):
    num2en = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
              "9": "nine", "0": "zero"}
    docs_source = []
    docs_target = []
    for i in range(10000):
        # 序列为长度在1到8之间的0-9的数字
        doc_len = random.randint(1, 8)
        doc_source = []
        doc_target = []
        for j in range(doc_len):
            num = str(random.randint(0, 9))
            doc_source.append(num)
            doc_target.append(num2en[num])
        docs_source.append(doc_source)
        docs_target.append(doc_target)

    return docs_source, docs_target

# 建立词汇索引表
# 针对当前任务即为，创建序列docs中所有数字和单词的索引
def make_vocab(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    for doc in docs:
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
    return w2i, i2w


def doc_to_seq(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    seqs = []
    for doc in docs:
        seq = []
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            seq.append(w2i[w])
        seqs.append(seq)
    return seqs, w2i, i2w

# 从样本集中抽取batch
# batch是序列的索引表示方式
def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size):
    ps = []
    # 生成用于抽取训练集的索引
    # ps是一个batch_size大小的list,每个数据是一个0-(训练集长度-1)的随机数,在训练中直接用该随机数作为索引抽取batch
    while len(ps) < batch_size:
        ps.append(random.randint(0, len(docs_source) - 1))

    source_batch = []
    target_batch = []

    # 所有随机的序列的长度的集合
    source_lens = [len(docs_source[p]) for p in ps]
    target_lens = [len(docs_target[p]) + 1 for p in ps]

    # 找出其中最长的那个
    max_source_len = max(source_lens)
    max_target_len = max(target_lens)

    for p in ps:
        # 按照所有可能的batch中最长的序列的长度将每个序列的末尾补上占位符对应的索引i
        source_seq = [w2i_source[w] for w in docs_source[p]] + \
                     [w2i_source["_PAD"]] * (max_source_len - len(docs_source[p]))
        # target不光要加上占位符还要加上EOS的符号表示结束
        target_seq = [w2i_target[w] for w in docs_target[p]] + \
                     [w2i_target["_PAD"]] * (max_target_len - 1 - len(docs_target[p])) + [w2i_target["_EOS"]]
        source_batch.append(source_seq)
        target_batch.append(target_seq)

    return source_batch, source_lens, target_batch, target_lens


if __name__ == "__main__":

    print("(1)load data......")
    docs_source, docs_target = load_data("")
    # source本代码中表示样本
    w2i_source, i2w_source = make_vocab(docs_source)
    # target本代码中表示label
    w2i_target, i2w_target = make_vocab(docs_target)

    print("(2) build model......")
    config = Config(w2i_source, w2i_target)
    # config.source_vocab_size = len(w2i_source)
    # config.target_vocab_size = len(w2i_target)
    model = Seq2seq(config=config, useTeacherForcing=True, useAttention=True, useBeamSearch=1)

    print("(3) run model......")
    batches = 3000
    # 每100个batches输出一次
    print_every = 100

    with tf.Session(config=tf_config) as sess:
        tf.summary.FileWriter('graph', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        losses = []
        total_loss = 0
        for batch in range(batches):
            source_batch, source_lens, target_batch, target_lens = get_batch(docs_source, w2i_source, docs_target,
                                                                             w2i_target, config.batch_size)

            feed_dict = {
                model.seq_inputs: source_batch,
                model.seq_inputs_length: source_lens,
                model.seq_targets: target_batch,
                model.seq_targets_length: target_lens
            }

            loss, _ = sess.run([model.loss, model.train_op], feed_dict)
            total_loss += loss

            # 每隔print_every个batch输出一次
            if batch % print_every == 0 and batch > 0:
                print_loss = total_loss if batch == 0 else total_loss / print_every
                losses.append(print_loss)
                total_loss = 0
                print("-----------------------------")
                print("batch:", batch, "/", batches)
                print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("loss:", print_loss)

                print("samples:\n")
                predict_batch = sess.run(model.out, feed_dict)
                for i in range(3):
                    print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"])
                    print("out:", [i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"])
                    print("tar:", [i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"])
                    print("")

        print(losses)
        print(saver.save(sess, "checkpoint/model.ckpt"))