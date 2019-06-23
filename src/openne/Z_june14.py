from __future__ import print_function
import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from .classify import Classifier, read_node_label


class _LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, negative_ratio=5, order=2):
        self.cur_epoch = 0
        self.order = order
        self.g = graph
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.gen_sampling_table() # negqtive sampling
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
        self.embeddings = tf.get_variable(name="embeddings"+str(self.order), shape=[
                                          self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.context_embeddings = tf.get_variable(name="context_embeddings"+str(self.order), shape=[
                                                  self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))

        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
        self.t_e_context = tf.nn.embedding_lookup(self.context_embeddings, self.t)
        self.second_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        self.first_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        if self.order == 1:
            self.loss = self.first_loss
        else:
            self.loss = self.second_loss
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(self.loss)

    def train_one_epoch(self):
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


    def ppr_walk(self, start_node, walk_length):
        '''
        Simulate a ppr walk starting from start node.
        '''
        G = self.g.G
        look_up = self.g.look_up_dict
        node_size = self.node_size

        walk = [start_node]
        alpha = 0.7
        while len(walk) < walk_length:
            if np.random.rand() < alpha:
                cur = walk[0]
            else:
                cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk


    def batch_iter(self):
        look_up = self.g.look_up_dict

        table_size = 1e8
        numNodes = self.node_size

        edges = [(look_up[x[0]], look_up[x[1]]) for x in self.g.G.edges()]

        data_size = self.g.G.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # window size = 10,  number of walks = 10
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        walk_length = 80
        walks = {}

        nodes = list(self.g.G.nodes())
        for node in nodes:
            walks[node] = self.ppr_walk(node, walk_length)

        for i in range(1, walk_length):
            for mod in range(mod_size):
                if mod == 0:
                    sign = 1.
                    h = []
                    t = []
                    cur_h = node
                    cur_t = walks[i]
                    h.append(cur_h)
                    t.append(cur_t)
                else:
                    sign = -1.
                    t = []
                    for i in range(len(h)):
                        t.append(
                            self.sampling_table[random.randint(0, table_size-1)])

                yield h, t, [sign]


    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        numNodes = self.node_size

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes)  # out degree

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes): # construct negative sampling table
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1
        '''
        data_size = self.g.G.number_of_edges()
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"]
                         for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"] *
                     data_size/total_sum for edge in self.g.G.edges()]
        # norm_prob = weight*size/sum, so it's may more than 1.

        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + \
                norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1
        '''

    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self.sess)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors


class Z(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, epoch=10, negative_ratio=5, order=3, label_file=None, clf_ratio=0.5, auto_save=True):
        self.rep_size = rep_size
        self.best_result = 0
        self.vectors = {}

        self.model = _LINE(graph, rep_size, batch_size,
                           negative_ratio)
        for i in range(epoch):
            self.model.train_one_epoch()
            if label_file:
                self.get_embeddings()
                X, Y = read_node_label(label_file)
                print("Training classifier using {:.2f}% nodes...".format(
                    clf_ratio*100))
                clf = Classifier(vectors=self.vectors,
                                 clf=LogisticRegression())
                result = clf.split_train_evaluate(X, Y, clf_ratio)

                if result['macro'] > self.best_result:
                    self.best_result = result['macro']
                    if auto_save:
                        self.best_vector = self.vectors

        self.get_embeddings()
        if auto_save and label_file:
            self.vectors = self.best_vector

    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = {}
        self.vectors = self.model.get_embeddings()

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
