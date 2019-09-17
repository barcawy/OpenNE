import networkx as nx
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
#from gensim.models import Word2Vec
from sklearn.cluster import KMeans

#from . import node2vec
#from . import Z_0623
from .graph import *

import warnings
warnings.filterwarnings('ignore')

def add_w(g):
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1.0
    return g

def construct_mapping_f_features_t_instances(g, k):
    G = g.G
    nodes = list(G.nodes())
    edges = list(G.edges())
    Centroid = [0] * k
    S = [0] * k
    for i in range(k):
        Centroid[i] = {}
        S[i] = []
    for node in nodes:
        Centroid[random.randint(0, k-1)][node] = 1
    for i in range(k):
        length = len(Centroid[i])
        Centroid[i] = {k:v/length for k,v in Centroid[i].items()}
    for i, edge in enumerate(edges):
        for c in range(k):
            if (edge[0] in Centroid[c].keys()) or (edge[1] in Centroid[c].keys()):
                S[c].append(i)

    return Centroid, S

def update_mapping_f_features_t_instances(g, k, Centroid):
    G = g.G
    nodes = list(G.nodes())
    edges = list(G.edges())

    S = [0] * k
    for i in range(k):
        S[i] = []

    for i, edge in enumerate(edges):
        for c in range(k):
            if (edge[0] in Centroid[c].keys()) or (edge[1] in Centroid[c].keys()):
                S[c].append(i)
    return S

def sim (edge, C):
    s = 0
    for e in edge:
        if e in C.keys():
            s += C[e]
    return s



def main():

    k = 20
    g = nx.read_gexf('../cluster_data/les-miserables.gexf')
    print(nx.info(g))

    nx_g = add_w(g)
    g = Graph()
    g.read_g(nx_g)
    nodes = list(g.G.nodes())
    edges = list(g.G.edges())
    MaxSim = [0.] * len(edges)
    CId = [0] * len(edges)
    Centroid, S = construct_mapping_f_features_t_instances(g, k)
    ite_num = 0
    counts = 1
    while(counts > 0):
        ite_num += 1
        counts = 0
        for i in range(k):
            S = update_mapping_f_features_t_instances(g, k ,Centroid)
            for s in S[i]:
                curSim = sim(edges[s], Centroid[i])
                if curSim > MaxSim[s]:
                    MaxSim[s] = curSim
                    CId[s] = i
                    counts += 1
        Centroid = [0] * k
        for i in range(k):
            Centroid[i] = {}
        for index, id in enumerate(CId):
            e0 = edges[index][0]
            e1 = edges[index][1]
            if e0 in Centroid[id].keys():
                Centroid[id][e0] += 1
            else:
                Centroid[id][e0] = 1
            if e1 in Centroid[id].keys():
                Centroid[id][e1] += 1
            else:
                Centroid[id][e1] = 1
        for i in range(k):
            length = len(Centroid[i])
            Centroid[i] = {k: v / length for k, v in Centroid[i].items()}
        print('Iteration %d: counts %d' % (ite_num, counts))

    l = 0
    for s in S:
        l += len(s)
    print(l)
    print('Done')


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main()

