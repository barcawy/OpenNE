import networkx as nx
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from . import node2vec
from . import Z_0623
from .graph import *

import warnings
warnings.filterwarnings('ignore')

def add_w(g):
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1.0
    return g

def main():
    g = nx.read_gexf('../cluster_data/les-miserables.gexf')
    print(nx.info(g))

    nx_g = add_w(g)
    g = Graph()
    g.read_g(nx_g)

    CN = 4
    # number of clusters
    P = 1
    # Return hyperparameter. Default is 1
    Q = 0.5
    # Inout hyperparameter. Default is 1.
    NW = 10
    # Number of walks per source. Default is 40
    WL = 40
    # Length of walk per source. Default is 10.
    WORKERS = 8
    # 'Number of parallel workers. Default is 8.
    WIN_SIZE = 10
    # Context size for optimization. Default is 10.
    ITER = 20
    # Number of epochs in SGD
    DIM = 16
    # Number of dimensions. Default is 128.
    PREFIX = 'les-miserables'
    n2v = False
    Pos = True
    if n2v:
        model = node2vec.Node2vec(graph=g, path_length=WL,
                                  num_paths=NW, dim=DIM,
                                  workers=WORKERS, p=P, q=Q, window=WIN_SIZE)
    else:
        model = Z_0623.Z(graph=g, path_length=WL,
                         num_paths=NW, dim=DIM, prefix=PREFIX,
                         workers=WORKERS, window=WIN_SIZE)

    kmeans_clustering = KMeans(n_clusters=CN)
    idx = kmeans_clustering.fit_predict(list(model.vectors.values()))

    g_draw = g.G.to_undirected()
    node_size = [8 * g_draw.degree(x) for x in g_draw.nodes() ]

    if Pos:
        with open("pos.file", "rb") as f:
            pos = pickle.load(f)
    else:
        pos = nx.spring_layout(g_draw)
        with open("pos.file", "wb") as f:
            pickle.dump(pos, f)

    nx.draw_networkx(g_draw, node_size = node_size, pos = pos, node_color = idx, with_labels=False)

    if n2v:
        plt.savefig('E:/Project/OpenNE/visualization/cv_n2v_' + str(CN) + '_' + str(P) +'_' + str(Q) + '.png')
    else:
        plt.savefig('E:/Project/OpenNE/visualization/cv_Z_d1.png')

    print('Done')


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main()

