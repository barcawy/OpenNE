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
        test_neg_file = os.path.join(args.dataset_dir, 'test.neg.txt.npy')
        test_neg_arr = np.load(open(test_neg_file, 'rb'))
        test_pos_file = os.path.join(args.dataset_dir, 'test.txt.npy')
        test_pos_arr = np.load(open(test_pos_file, 'rb'))

        train_pos_file = os.path.join(args.dataset_dir, 'train.txt.npy')
        train_neg_file = os.path.join(args.dataset_dir, 'train.neg.txt.npy')
        train_pos_arr = np.load(open(train_pos_file, 'rb'))
        train_neg_arr = np.load(open(train_neg_file, 'rb'))

        pos_train_edge_embs = get_edge_embeddings(vectors, train_pos_arr)
        neg_train_edge_embs = get_edge_embeddings(vectors, train_neg_arr)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_pos_arr)), np.zeros(len(train_neg_arr))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(vectors, test_pos_arr)
        neg_test_edge_embs = get_edge_embeddings(vectors, test_neg_arr)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_pos_arr)), np.zeros(len(test_neg_arr))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        test_roc = roc_auc_score(test_edge_labels, test_preds)
        test_prec = average_precision_score(test_edge_labels, test_preds)
        print('Method: %s, ROC: %f, Prec: %f, time: %d' % (args.method, test_roc, test_prec, t2 - t1))


    elif args.label_file and args.method == 'c':
        X, Y = read_node_label(args.label_file)  # groupid list
        # X, Y = read_node_label_index(args.label_file)  # 单列groupid

        # ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # for ratio in ratios:
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        results = clf.split_train_evaluate(X, Y, args.clf_ratio)

if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())


