from __future__ import print_function
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
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
from .classify import Classifier, read_node_label, read_node_label_index
from . import line
from . import tadw
from .gcn import gcnAPI
from . import lle
from . import hope
from . import lap
from . import gf
from . import sdne
from .grarep import GraRep
from . import Mayten
import time
import ast

from . import cll

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--dataset-dir', required=True,
                        help='Input graph file')
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
        'mayten'
    ], help='The learning method')

    args = parser.parse_args()
    return args


IS_DIRECTED = None


def IsDirected(dataset_dir):
  global IS_DIRECTED
  if IS_DIRECTED is not None:
    return IS_DIRECTED
  IS_DIRECTED = os.path.exists(
      os.path.join(dataset_dir, 'test.directed.neg.txt.npy'))
  return IS_DIRECTED


def GetNumNodes(dataset_dir):
  global NUM_NODES
  if NUM_NODES == 0:
    index = pickle.load(
        open(os.path.join(dataset_dir, 'index.pkl'), 'rb'))
    NUM_NODES = len(index['index'])
  return NUM_NODES


def GetOrMakeAdjacencyMatrix(dataset_dir):
  """Creates Adjacency matrix and caches it on disk with name a.npy."""
  a_file = os.path.join(dataset_dir, 'a.npy')
  if os.path.exists(a_file):
    return np.load(open(a_file, 'rb'))

  num_nodes = GetNumNodes()
  a = np.zeros(shape=(num_nodes, num_nodes), dtype='float32')
  train_edges = np.load(
      open(os.path.join(dataset_dir, 'train.txt.npy'), 'rb'))
  a[train_edges[:, 0], train_edges[:, 1]] = 1.0
  if not IsDirected():
    a[train_edges[:, 1], train_edges[:, 0]] = 1.0

  np.save(open(a_file, 'wb'), a)
  return a


def main(args):
    t1 = time.time()

    if IsDirected(args.dataset-dir):
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
    g.read_adjlist(filename=train_pos_file)

    if args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'deepWalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'mayten':
        model = Mayten.Mayten(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size)

    t2 = time.time()
    print('time: %d \n' %(t2 - t1))

    print("Saving embeddings...")
    model.save_embeddings(args.output)
    vectors = model.vectors


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())