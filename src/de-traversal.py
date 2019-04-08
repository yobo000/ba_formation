# -*- coding: utf-8 -*-
"""
Linkcut with Traversal
Traversal all the edges
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
from operator import itemgetter
import logging

from networkx.utils import py_random_state
from networkx.generators.classic import empty_graph

N = 1000  # 10000
# edge pramas 3
M = 3  # 3
# growth node number 50
M_0 = 50
# iter times 10
T = 10
# perferential attachment tolerant
THERSHOLD = 0.3
# deffuant tolerant
TOLERANT = 0.3
DEFFUANT_COEFF = 0.5
PRECISION = 0.00001  # 临时使用, 由于random_sample只在(0,1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename="output.log")
logger = logging.getLogger(__name__)


def opinion_thershold(opinion):
    max_value = opinion + THERSHOLD
    min_value = opinion - THERSHOLD
    end = 1 if max_value > 1 else max_value
    start = 0 if min_value < 0 else min_value
    return start, end


def opinion_filter(opinion, pa_nodes):
    start, end = opinion_thershold(opinion)
    targets = list(filter(
        lambda x: x[1]["opinion"] < end and x[1]["opinion"] > start,
        pa_nodes))
    return list(map(lambda x: x[0], targets))


def _random_subset(seq, m, rng):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    : Keep It!!!
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


class Diss_Network(object):
    """docstring for Dis_Network"""
    def __init__(self, n, m):
        self.number = n
        self.growth = m
        self.graph = empty_graph(m)

    def save_degree_distribution(self):
        degree_sequence = sorted([d for n, d in self.graph.degree()],
                                 reverse=True)
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        deg = np.array(deg)
        cnt = np.array(cnt)
        connectivity = np.divide(cnt, N)
        deg_log = deg
        deg_cnt = connectivity
        order = np.argsort(deg_log)
        deg_log_array = np.array(deg_log)[order]
        deg_cnt_array = np.array(deg_cnt)[order]
        plt.loglog(deg_log_array, deg_cnt_array, ".")
        # plt.show()
        plt.savefig('result1_trav.png')

    def save_opinion_distribution(self):
        opinions = np.array(
            list(nx.get_node_attributes(self.graph, 'opinion').values()))
        # hist_data = np.histogram(opinions.values())
        bins = [0.01 * n for n in range(100)]
        plt.hist(opinions, bins=bins)
        plt.title("Histogram of opinions")
        # plt.show()
        plt.savefig('result2_trav.png')

    def save_distribution(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        deg = np.array(deg)
        cnt = np.array(cnt)
        connectivity = np.divide(cnt, N)
        deg_log = deg  # np.log10(deg)
        deg_cnt = connectivity  # np.log10(connectivity)
        order = np.argsort(deg_log)
        deg_log_array = np.array(deg_log)[order]
        deg_cnt_array = np.array(deg_cnt)[order]
        ax1.loglog(deg_log_array, deg_cnt_array, ".")
        ax1.set_ylim(0, 5)

        opinions = np.array(list(nx.get_node_attributes(self.graph, 'opinion').values()))
        # hist_data = np.histogram(opinions.values())
        bins = [0.01 * n for n in range(100)]
        ax2.hist(opinions, bins=bins)

        plt.savefig('result.png')

    @py_random_state(1)
    def barabasi_albert_with_opinion_graph(self, seed=None):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.
        """
        if self.growth < 1 or self.growth >= self.number:
            raise nx.NetworkXError("Barabási–Albert network must have m < n")
        # Add m initial nodes (m0 in barabasi-speak)
        s = np.random.random_sample(self.growth)
        # print("initial value:", dict(enumerate(s)))
        nx.set_node_attributes(self.graph, dict(enumerate(s)), 'opinion')
        # Target nodes for new edges
        targets = list(range(self.growth))
        # List of existing nodes
        # with nodes repeated once for each adjacent edge
        repeated_nodes = []
        # Start adding the other n-m nodes. The first node is m.
        source = self.growth
        while source < self.number:
            opinion_value = np.random.random_sample()
            # Filter nodes from the targets
            nodes = list(self.graph.nodes(data=True))
            pa_nodes = itemgetter(*targets)(nodes)
            targets = opinion_filter(opinion_value, pa_nodes)
            # Add node with opinion
            self.graph.add_node(source, opinion=opinion_value)
            # Add edges to m nodes from the source.
            self.graph.add_edges_from(zip([source] * self.growth, targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source] * self.growth)
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = _random_subset(repeated_nodes, self.growth, seed)
            source += 1
        return self.graph

    def formation(self, node1, node2):
        """
        value = node1.value + node2.value
        """
        value1 = self.graph.node[node1]['opinion']
        value2 = self.graph.node[node2]['opinion']
        diff = abs(value1 - value2)
        if diff < TOLERANT and diff > PRECISION:
            value_1 = value1 - DEFFUANT_COEFF * (value1 - value2)
            value_2 = value2 - DEFFUANT_COEFF * (value2 - value1)
            return value_1, value_2
        elif diff < PRECISION:
            return True, False
        else:
            return False, False

    def opinion_formation(self):
        """
        1. 取全部edges
        2. 挨个做formation
        3. 重复直到收敛
        在容忍度内, 取均值，容忍度外 断开edge
        """
        counter = 0
        not_convergence = True
        while not_convergence:
            # G.edges 返回一个有两个node编号的tuple
            graph_edges = list(self.graph.edges(data=False))
            if graph_edges:
                # random edges
                edge_len = len(graph_edges)
                for edge in graph_edges:
                    value1, value2 = self.formation(*edge)
                    if value2:
                        node1, node2 = edge
                        self.graph.node[node1]['opinion'] = value1
                        self.graph.node[node2]['opinion'] = value2
                    elif value1:
                        counter += 1
                    else:
                        # remove edge
                        self.graph.remove_edge(*edge)
                if counter * 1.5 > edge_len:
                    not_convergence = False
                else:
                    counter = 0
            else:
                not_convergence = False

    def get_nodes_length(self):
        return nx.number_of_nodes(self.graph)

    def get_edges_length(self):
        return nx.number_of_edges(self.graph)


if __name__ == '__main__':
    network = Diss_Network(N, M)
    network.barabasi_albert_with_opinion_graph()
    print("Nodes: ", network.get_nodes_length())
    print("Edges: ", network.get_edges_length())
    network.opinion_formation()
    network.save_distribution()
