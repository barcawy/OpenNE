from __future__ import print_function
import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import os
import pickle

import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .graph import *
from . import node2vec
from . import line
from . import tadw
from .gcn import gcnAPI
from . import lle
from . import hope
from . import lap
from . import gf
from . import sdne
from . import app
from . import verse
from .grarep import GraRep
from . import Z_mayten
from . import Z_0623
from . import Z_0709
import time
import ast

from . import cll

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--dataset-dir', required=True,
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--prefix',
                        help='dataset prefix')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--hop', default = 1, type=int,
                        help='The most hop in subGraph')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=[
        'node2vec',
        'deepWalk',
        'line',
        'gcn',
        'grarep',
        'tadw',
        'lle',
        'hope',
        'lap',
        'gf',
        'sdne',
        'cll',
        'mayten',
        'app',
        'verse'
    ], help='The learning method')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')

    args = parser.parse_args()
    return args

def Generate_test(g):
    nodes = list(g.G.nodes())
    count = 1000
    test_arr = {}
    for node in nodes:
        test_arr[node] = []
    for node in nodes:
        degree = g.G.out_degree(node)
        if degree > 9:
            count -= 1
            if count == 0:
                return g, test_arr
            k = int(0.2 * degree)
            neigh = list(g.G.neighbors(node))
            #print(node, neigh)
            index = random.sample(neigh, k)
            #print(index)
            for i in index:
                if i == node:
                    continue
                g.G.remove_edge(node, i)
                g.G.remove_edge(i, node)
                test_arr[node].append(i)
    return g ,test_arr

def main(args):
    t1 = time.time()
    test_pos_file = os.path.join(args.dataset_dir, 'test.txt.npy')
    test_pos_arr = np.load(open(test_pos_file, 'rb'))

    train_pos_file = os.path.join(args.dataset_dir, 'train.txt.npy')
    #train_neg_file = os.path.join(args.dataset_dir, 'train.neg.txt.npy')
    train_pos_arr = np.load(open(train_pos_file, 'rb'))
    #train_neg_arr = np.load(open(train_neg_file, 'rb'))
    pos_arr = np.row_stack((train_pos_arr, test_pos_arr))
    g = Graph()
    # g.read_npy(edgelist=train_pos_arr, directed=args.directed) word2vec ERROR
    np.savetxt('data.txt', pos_arr, fmt='%d')
    g.read_edgelist('data.txt', weighted=args.weighted, directed=args.directed)
    g ,test_arr= Generate_test(g)
    if args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'deepWalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'mayten':
        model = Z_0709.Z(graph=g, path_length=args.walk_length,
                                num_paths=args.number_walks, dim=args.representation_size, prefix = args.prefix,
                                hop = args.hop, workers=args.workers, window=args.window_size)
    elif args.method == 'grarep':
        model = GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)
    elif args.method == 'line':
        model = line.LINE(g, epoch=args.epochs,
                      rep_size=args.representation_size, order=args.order)
    elif args.method == 'app':
        model = app.App(g, epoch=args.epochs)
    elif args.method == 'verse':
        model = verse.verse(g, epoch=args.epochs)
    t2 = time.time()
    # print('time: %d \n' %(t2 - t1))

    print("Saving embeddings...")
    # model.save_embeddings(args.output)

    ks = [20,50,100] # top 20 50 100
    for k in ks:
        abcd = [-1000] * k
        predict_count = k * 1000
        correct_count = 0
        test_count = 0
        if args.method == 'app':
            vectors_s = model.vectors_s
            vectors_t = model.vectors_t
            nodes = list(g.G.nodes())
            for key in test_arr.keys():
                if len(test_arr[key]):
                    test_count += len(test_arr[key])
                    d = {}
                    l = []
                    for node in nodes:
                        # app
                        #edge_value = np.dot(vectors_s[key], vectors_t[node])
                        # ASE
                        edge_value = np.dot(vectors_s[key], vectors_t[node]) + np.dot(vectors_s[node], vectors_t[key])
                        d[node] = edge_value
                        l.append(edge_value)
                    kaxian = sorted(l, reverse=True)[k]
                    for node in d.keys():
                        if d[node] >= kaxian:
                            if node in test_arr[key]:
                                correct_count += 1
                    #print('Node: %s, Prec: %f, Recall: %f, K: %d' % (key, float(correct_count) / predict_count,
                     #                                               float(correct_count) / test_count, k))
            print('Method: %s, Prec: %f, Recall: %f, K: %d, time: %d' %(args.method, float(correct_count) / predict_count,
                                                                    float(correct_count) / test_count, k, t2 - t1))
        else:
            vectors = model.vectors
            nodes = list(g.G.nodes())
            for key in test_arr.keys():
                if len(test_arr[key]):
                    test_count += len(test_arr[key])
                    d = {}
                    l = []
                    for node in nodes:
                        edge_value =np.dot(vectors[key], vectors[node])
                        d[node] = edge_value
                        l.append(edge_value)
                    kaxian = sorted(l, reverse=True)[k]
                    for node in d.keys():
                        if d[node] >= kaxian:
                            if node in test_arr[key]:
                                correct_count += 1
                    #print('Node: %s, Prec: %f, Recall: %f, K: %d' % (key, float(correct_count) / predict_count,
                     #                                                float(correct_count) / test_count, k))
            print('Method: %s, Prec: %f, Recall: %f, K: %d, time: %d' %(args.method, float(correct_count)/predict_count,
                                                                        float(correct_count)/test_count, k, t2-t1))




if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())