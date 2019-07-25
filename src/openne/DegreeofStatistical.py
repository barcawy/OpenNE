from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .graph import *

def main():
    g = Graph()
    print("Reading...")

    # args
    input = '../data/blogCatalog/bc_edgelist.txt '
    graph_format = 'edgelist'
    weighted = False
    directed = False

    if graph_format == 'adjlist':
        g.read_adjlist(filename=input)
    elif graph_format == 'edgelist':
        g.read_edgelist(filename=input, weighted=weighted,
                        directed=directed)

    duideD = []
    cuodeD = []
    file = open('duide.txt', 'r')
    duide = file.readlines()
    for d in duide:
        d = d.strip('\n')
        duideD.append(g.G.degree(d)/2)
    file.close()
    file = open('cuode.txt', 'r')
    cuode = file.readlines()
    for d in cuode:
        d = d.strip('\n')
        cuodeD.append(g.G.degree(d)/2)

    dl = [1,2,5,10,30,50,100,1e5]
    dT = [0, 0, 0, 0, 0, 0, 0, 0]
    cT = [0, 0, 0, 0, 0, 0, 0, 0]

    for d in duideD:
        i = 0
        while d > dl[i]:
            i += 1
        dT[i] += 1
    for c in cuodeD:
        i = 0
        while c > dl[i]:
            i += 1
        cT[i] += 1
    print(dT)
    print(cT)
    print(sum(dT)/sum(dT+cT))
    print([a / (a + b + 1e-8) for a, b in zip(dT, cT)])


if __name__ == "__main__":
    main()