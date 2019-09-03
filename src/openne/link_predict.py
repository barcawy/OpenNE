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


def get_edge_embeddings(emb_matrix, edge_list):
    embs = []
    for edge in edge_list:
        emb1 = emb_matrix[str(edge[0])]
        emb2 = emb_matrix[str(edge[1])]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

def get_dire_edge_embeddings(g, emb_matrix_s, emb_matrix_t, edge_list, directed):
    # V 2.0
    if not directed:
        embs = []
        for edge in edge_list:
            emb_0s = emb_matrix_s[str(edge[0])]
            emb_1s = emb_matrix_s[str(edge[1])]
            emb_0t = emb_matrix_t[str(edge[0])]
            emb_1t = emb_matrix_t[str(edge[1])]
            edge_emb = np.dot(emb_0s, emb_1t) + np.dot(emb_1s, emb_0t)
            embs.append(edge_emb.reshape(-1))
        embs = np.array(embs)
        return embs
    else:# 0.8962
        # V 1.0
        embs = []
        for edge in edge_list:
            emb1 = emb_matrix_s[str(edge[0])]
            emb2 = emb_matrix_t[str(edge[1])]
            edge_emb = np.dot(emb1, emb2)
            embs.append(edge_emb.reshape(-1))
        embs = np.array(embs)
        return embs


def get_degree(g, edge_list):
    degrees = {1:[], 2:[], 5:[], 10:[], 30:[], 50:[], 100:[], 1e5:[]}
    dl = [1, 2, 5, 10, 30, 50, 100, 1e5]
    for index, edge in enumerate(edge_list):
        degree1 = g.degree(str(edge[0]))
        degree2 = g.degree(str(edge[1]))
        min_d = min(degree1, degree2)
        i = 0
        while(min_d > dl[i]):
            i += 1
        degrees[dl[i]].append(index)
    return degrees

def doS(g, vectors, edge_classifier,test_pos_arr, test_neg_arr, pos_test_edge_embs, neg_test_edge_embs):
    pos_degree_dict = get_degree(g, test_pos_arr)
    neg_degree_dict = get_degree(g, test_neg_arr)
    for key in pos_degree_dict.keys():
        pos_embs = pos_test_edge_embs[pos_degree_dict[key]]
        neg_embs = neg_test_edge_embs[neg_degree_dict[key]]
        if (len(pos_degree_dict[key]) < 10) or (len(neg_degree_dict[key]) < 10):
            print('data poor')
            continue
        test_embs = np.concatenate([pos_embs, neg_embs])
        test_labels = np.concatenate([np.ones(len(pos_embs)), np.zeros(len(neg_embs))])
        test_preds = edge_classifier.predict_proba(test_embs)[:, 1]
        test_roc = roc_auc_score(test_labels, test_preds)
        test_prec = average_precision_score(test_labels, test_preds)
        print('Degree: %s, ROC: %f, Prec: %f' % (key, test_roc, test_prec))


def main(args):
    t1 = time.time()

    if args.directed:
        test_neg_file = os.path.join(args.dataset_dir, 'test.directed.neg.txt.npy')
        test_neg_arr = np.load(open(test_neg_file, 'rb'))
    else:
        test_neg_file = os.path.join(args.dataset_dir, 'test.neg.txt.npy')
        test_neg_arr = np.load(open(test_neg_file, 'rb'))
    test_pos_file = os.path.join(args.dataset_dir, 'test.txt.npy')
    test_pos_arr = np.load(open(test_pos_file, 'rb'))

    train_pos_file = os.path.join(args.dataset_dir, 'train.txt.npy')
    train_neg_file = os.path.join(args.dataset_dir, 'train.neg.txt.npy')
    train_pos_arr = np.load(open(train_pos_file, 'rb'))
    train_neg_arr = np.load(open(train_neg_file, 'rb'))

    g = Graph()
    # g.read_npy(edgelist=train_pos_arr, directed=args.directed) word2vec ERROR
    np.savetxt('data.txt', train_pos_arr, fmt='%d')
    g.read_edgelist('data.txt', weighted=args.weighted, directed=args.directed)

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

    if args.method == 'app':
        vectors_s = model.vectors_s
        vectors_t = model.vectors_t
        pos_train_edge_embs = get_dire_edge_embeddings(g, vectors_s, vectors_t, train_pos_arr, directed=args.directed)
        neg_train_edge_embs = get_dire_edge_embeddings(g, vectors_s, vectors_t, train_neg_arr, directed=args.directed)
    else:
        vectors = model.vectors
        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(vectors, train_pos_arr)
        neg_train_edge_embs = get_edge_embeddings(vectors, train_neg_arr)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_pos_arr)), np.zeros(len(train_neg_arr))])

    # Test-set edge embeddings, labels
    if args.method == 'app':
        pos_test_edge_embs = get_dire_edge_embeddings(g, vectors_s, vectors_t, test_pos_arr, directed=args.directed)
        neg_test_edge_embs = get_dire_edge_embeddings(g, vectors_s, vectors_t, test_neg_arr, directed=args.directed)
    else:
        pos_test_edge_embs = get_edge_embeddings(vectors, test_pos_arr)
        neg_test_edge_embs = get_edge_embeddings(vectors, test_neg_arr)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create test-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_pos_arr)), np.zeros(len(test_neg_arr))])

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_roc = roc_auc_score(test_edge_labels, test_preds)
    test_prec = average_precision_score(test_edge_labels, test_preds)
    print('Method: %s, ROC: %f, Prec: %f, time: %d' %(args.method, test_roc, test_prec, t2-t1))

    # doS(g.G, vectors, edge_classifier,test_pos_arr, test_neg_arr, pos_test_edge_embs, neg_test_edge_embs)


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())