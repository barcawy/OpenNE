from __future__ import print_function
import time
import math
import random
import numpy as np
import pickle as pkl
import networkx as nx
from gensim.models import Word2Vec
from fastdtw import fastdtw
from collections import Counter
from collections import defaultdict
import os

class Z(object):

    def __init__(self, graph, path_length, num_paths, dim, prefix, hop,  **kwargs):

        kwargs["workers"] = kwargs.get("workers", 4)

        #kwargs["hs"] = 1 # 1 分层softmax 0 负采样

        self.graph = graph
        preprocess = False
        if preprocess:
            self.ppr_matrix = self.constructSubGraph(hop)
            self.degrees, self.degree_permuted = self.create_degree()
            self.degree_neighbors, self.norm_weight = self.create_ppr_sample_table()
            self.dump_to_disk(self.degree_neighbors,'E:/Project/OpenNE/matrix_pkl/' + prefix + '_'+ str(hop) + '_neighbors')
            self.dump_to_disk(self.norm_weight,'E:/Project/OpenNE/matrix_pkl/' + prefix + '_'+ str(hop) + '_weight')
        else:
            self.degree_neighbors = self.load_pkl('E:/Project/OpenNE/matrix_pkl/' + prefix + '_'+ str(hop) + '_neighbors')
            self.norm_weight = self.load_pkl('E:/Project/OpenNE/matrix_pkl/' + prefix + '_'+ str(hop) + '_weight')
        sentences = self.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1 # 1 skipgram; 0 CBOW

        self.size = kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def dump_to_disk(self, f, file_name):
        with open(file_name + '.pkl', 'wb') as handle:
            pkl.dump(f, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_pkl(self, file_name):
        with open(file_name + '.pkl', 'rb') as handle:
            val = pkl.load(handle)
        return val

    def neighbors(self, fringe):
        # find all 1-hop neighbors of nodes in fringe from A
        graph = self.graph.G
        res = set()
        for node in fringe:
            nei = graph.neighbors(node)
            nei = set(nei)
            res = res.union(nei)
        return res

    def constructSubGraph(self, hop):
        graph = self.graph.G
        edge_set = set(graph.edges())
        nodes = list(graph.nodes())
        #subgraph_map = defaultdict(nx.Graph)
        ppr_matrix = {}
        for node in nodes:
            subgraph_map = nx.Graph()
            subgraph_map.add_node(node)
            fringe = set(node)
            visited = set(node)
            for dist in range(0, hop):
                fringe = self.neighbors(fringe)
                fringe = fringe - visited
                visited = visited.union(fringe)
            visited = list(visited)
            for pos_u, u in enumerate(visited):
                for v in visited[pos_u+1:]:
                    if (u, v) in edge_set or (v, u) in edge_set:
                        subgraph_map.add_edge(u, v)

            ppr_matrix[node] = Counter()
            walk = self.subgraph_walk(subgraph_map, walk_length=500, start_node=node)
            ppr_matrix[node].update(walk)
        return ppr_matrix

    def subgraph_walk(self, subGraph, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = subGraph
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                # 独立的点
                break
        return walk

    def deepwalk_walk(self, walk_length, start_node, alpha = 0.5):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.graph.G
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            alpha = 1#alpha/G.degree(cur)
            if np.random.rand() < alpha:
                walk.append(np.random.choice(self.degree_neighbors[cur], p=self.norm_weight[cur]))
            else:
                cur_nbrs = list(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    # node2vec
                    n2v = 0
                    if n2v:
                        nbr = random.choice(cur_nbrs)
                        if set(cur_nbrs) & set(G.neighbors(nbr)):
                            walk.append(random.choice(cur_nbrs))
                        else:
                            walk.append(nbr)
                    else:
                    # deepwalk
                        walk.append(random.choice(cur_nbrs))
                else:
                    break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.graph.G
        walks = []
        nodes = list(G.nodes())
        print('Simulate walk iteration:')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print(len(walks))
        return walks

    def create_degree(self):
        G = self.graph.G
        print("- Creating degree vectors...")
        degrees = {}
        degrees_sorted = set()
        degree_permuted = {}
        for v in G.nodes():
            degree = G.degree(v)
            degrees_sorted.add(degree)
            degree_permuted[v] = degree
            if (degree not in degrees):
                degrees[degree] = {}
                degrees[degree]['vertices'] = []
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        # degree_permuted = degrees_sorted
        degrees_sorted = np.sort(degrees_sorted)
        l = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if (index > 0):
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if (index < (l - 1)):
                degrees[degree]['after'] = degrees_sorted[index + 1]
        print("- Degree vectors created.")
        return degrees, degree_permuted

    def create_ppr_sample_table(self):
        print("- Creating PPR sample table ...")
        nodes = list(self.graph.G.nodes())
        degree_neighbors = {}
        norm_weight = {}
        nodes_num = len(nodes)
        k = 0
        for node in nodes:
            print(str(k + 1), '/', str(nodes_num))
            k += 1
            degree_neighbors[node] = self.get_vertices(node)
            norm_weight[node] = self.ppr_sample(node, degree_neighbors[node])
        print("- PPR sample table created.")
        return degree_neighbors, norm_weight

    def cost(self, a, b):
        ep = 0.001
        m = max(a, b) + ep
        mi = min(a, b) + ep
        return ((m / mi) - 1)

    def ppr_sample(self, node, neighbors):
        node_ppr_v = [i[1] for i in self.ppr_matrix[node].most_common()]#[1:]
        if len(node_ppr_v) == 0:
            node_ppr_v = [1]
        sim_list = []
        nodes_num = len(self.graph.G.nodes())
        for _neighbor in neighbors:
            neighbor_ppr_v = [i[1] for i in self.ppr_matrix[_neighbor].most_common()]#[1:]
            if len(neighbor_ppr_v) == 0:
                neighbor_ppr_v = [1]
            dits_dtw, _ = fastdtw(node_ppr_v, neighbor_ppr_v, radius=1, dist=self.cost)
            sim_list.append(np.exp(-1.0 * dits_dtw))

        norm_weight = [float(i) / sum(sim_list) for i in sim_list]
        # sampled_neighbor = np.random.choice(neighbors, p=norm_weight)
        return norm_weight

    def verifyDegrees(self, degree_v_root, degree_a, degree_b):

        if (degree_b == -1):
            degree_now = degree_a
        elif (degree_a == -1):
            degree_now = degree_b
        elif (abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
            degree_now = degree_b
        else:
            degree_now = degree_a

        return degree_now

    def get_vertices(self, v):
        num_seleted = 2 * math.log(len(self.graph.G.nodes()), 2)
        vertices = []

        degree_v = self.graph.G.degree(v)

        try:
            c_v = 0

            for v2 in self.degrees[degree_v]['vertices']:
                if (v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if (c_v > num_seleted):
                        raise StopIteration

            if ('before' not in self.degrees[degree_v]):
                degree_b = -1
            else:
                degree_b = self.degrees[degree_v]['before']
            if ('after' not in self.degrees[degree_v]):
                degree_a = -1
            else:
                degree_a = self.degrees[degree_v]['after']
            if (degree_b == -1 and degree_a == -1):
                raise StopIteration
            degree_now = self.verifyDegrees(degree_v, degree_a, degree_b)

            while True:
                for v2 in self.degrees[degree_now]['vertices']:
                    if (v != v2):
                        vertices.append(v2)
                        c_v += 1
                        if (c_v > num_seleted):
                            raise StopIteration

                if (degree_now == degree_b):
                    if ('before' not in self.degrees[degree_b]):
                        degree_b = -1
                    else:
                        degree_b = self.degrees[degree_b]['before']
                else:
                    if ('after' not in self.degrees[degree_a]):
                        degree_a = -1
                    else:
                        degree_a = self.degrees[degree_a]['after']

                if (degree_b == -1 and degree_a == -1):
                    raise StopIteration

                degree_now = self.verifyDegrees(degree_v, degree_a, degree_b)

        except StopIteration:
            return list(vertices)

        return list(vertices)

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

