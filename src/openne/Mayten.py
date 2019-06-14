from __future__ import print_function
import time
import random
import numpy as np
from gensim.models import Word2Vec


class Mayten(object):

    def __init__(self, graph, path_length, num_paths, dim, **kwargs):
        kwargs["workers"] = kwargs.get("workers", 1)
        self.graph = graph
        self.walker = Surfing(graph, workers=kwargs["workers"])
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

    def save_results(self, filename, method, ratio, result):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {} {} \n".format(method, ratio, result))
        fout.close()


class Surfing(object):
    def __init__(self, G, workers):
        self.G = G.G
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def deepwalk_walk(self, walk_length, start_node, alpha):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        walk = [start_node]
        cur = walk[-1]
        while len(walk) < walk_length:
            if np.random.rand() < alpha[look_up_dict[start_node]]:
                cur = walk[-1]
            else:
                cur = walk[0]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def get_median(self, data):
        data.sort()
        half = len(data) // 2
        return (data[half] + data[~half]) / 2

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        alpha = [0] * len(nodes)
        look_up_dict = self.look_up_dict
        for node in nodes:
            alpha[look_up_dict[node]] = G.out_degree(node)
        half = self.get_median(alpha)
        alpha = [a / (a + half) for a in alpha]
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node, alpha = alpha))
            # pool.close()
            # pool.join()
        # print(len(walks))
        return walks

