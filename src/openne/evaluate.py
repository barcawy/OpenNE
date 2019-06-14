from __future__ import print_function

import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression

from .link_predict import *
from .classify import Classifier, read_node_label, read_node_label_index

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--emb-file', required=True,
                        help='Input embedding file')
    parser.add_argument('--method', required=True, choices=[
        'l',
        'c'
    ], help='Evaluate Method: Link prediction, Classification')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    args = parser.parse_args()
    return args


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def main(args):
    vectors = load_embeddings(args.emb)
    if args.method == 'l':
        print()
    elif args.label_file and args.method == 'c':
        X, Y = read_node_label(args.label_file)  # groupid list
        # X, Y = read_node_label_index(args.label_file)  # 单列groupid

        # ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # for ratio in ratios:
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        results = clf.split_train_evaluate(X, Y, args.clf_ratio)




