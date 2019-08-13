# Scalable Graph Embedding for Asymmetric Proximity ,AAAI, 2017
from __future__ import print_function
import random
import math
import numpy as np
import tensorflow as tf


class _LINE(object):

    def __init__(self, graph, rep_size = 128, n_sample = 200, max_step = 10, negative_ratio = 5):
        self.cur_epoch = 0
        self.g = graph
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.n_sample = n_sample
        self.max_step = max_step
        self.negative_ratio = negative_ratio
        self.batch_size = 1000
        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer( # init weight
            uniform=False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])

        cur_seed = random.getrandbits(32)
        self.embeddings = tf.get_variable(name="embeddings", shape=[
                                          self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.context_embeddings = tf.get_variable(name="context_embeddings", shape=[
                                                  self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        self.t_e_context = tf.nn.embedding_lookup(self.context_embeddings, self.t)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(self.loss)

    def walker(self, start_node, max_step = 10, alpha = 0.85):

        G = self.g.G
        walk = [start_node]
        while len(walk) < max_step:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if np.random.rand() < alpha:
                if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
                else:
                    break
            else:
                break
        return walk[-1]

    def train_one_epoch(self):
        # 原coding是 epoch*sample=200*200=40000
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            feed_dict = {
                self.h: h,
                self.t: t,
                self.sign: sign,
            }
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict)
            sum_loss += cur_loss
            batch_id += 1
        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1

    def batch_iter(self):
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        numNodes = self.node_size

        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0

        start_index = 0
        end_index = min(start_index+self.batch_size, numNodes)
        while start_index < numNodes:
            if mod == 0:
                sign = 1.
                h = []
                t = []
                for i in range(start_index, end_index):
                    ppr_node = self.walker(look_back[i])
                    cur_h = i
                    cur_t = look_up[ppr_node]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(h)):
                    t.append(random.randint(0, numNodes-1))

            yield h, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index+self.batch_size, numNodes)

    def get_embeddings(self):
        vectors_s = {}
        vectors_t = {}
        embeddings = self.embeddings.eval(session=self.sess)
        context_embeddings = self.context_embeddings.eval(session=self.sess)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors_s[look_back[i]] = embedding
        for i, embedding in enumerate(context_embeddings):
            vectors_t[look_back[i]] = embedding
        return vectors_s, vectors_t


class App(object):

    def __init__(self, graph, rep_size=128, n_sample=3, max_step=10, epoch = 50, negative_ratio=3):
        self.rep_size = rep_size
        self.model = _LINE(graph, rep_size, n_sample, max_step, negative_ratio)
        for i in range(epoch):
            self.model.train_one_epoch()

        self.get_embeddings()

    def get_embeddings(self):
        ## self.last_vectors = self.vectors
        self.vectors_s = {}
        self.vectors_t = {}
        self.vectors_s, self.vectors_t = self.model.get_embeddings()

    def save_embeddings(self, filename):
        #需要重写双向量保存的代码
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
