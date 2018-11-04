# -*- coding: utf-8 -*-
"""
    generate ab graph with data
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
from operator import itemgetter

from networkx.utils import py_random_state
from networkx.generators.classic import empty_graph

N = 10000
# edge pramas
M = 3
# growth node number
M_0 = 50
# iter times
T = 10
# deffuant tolerant
TOLERANT = 0.3


def show_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
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
    plt.loglog(deg_log_array, deg_cnt_array, ".")
    plt.show()


def deffuant_value(opinion):
    max_value = opinion + TOLERANT
    min_value = opinion - TOLERANT
    end = 1 if max_value > 1 else max_value
    start = 0 if min_value < 0 else min_value
    return start, end


def opinion_filter(opinion, pa_nodes):
    start, end = deffuant_value(opinion)
    targets = list(filter(
        lambda x: x[1]["opinion"] < end and x[1]["opinion"] > start, pa_nodes))
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


@py_random_state(2)
def barabasi_albert_with_opinion_graph(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.
    """

    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m)
    # Set opinion and filter targets
    opinion = True
    if opinion:
        s = np.random.random_sample(m)
        # print("initial value:", s)
        nx.set_node_attributes(G, dict(enumerate(s)), 'opinion')
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        if opinion:
            opinion_value = np.random.random_sample()
            # Filter nodes from the targets
            nodes = list(G.nodes(data=True))
            pa_nodes = itemgetter(*targets)(nodes)
            targets = opinion_filter(opinion_value, pa_nodes)
            # Add node with opinion
            G.add_node(source, opinion=opinion_value)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G


if __name__ == '__main__':
    G = barabasi_albert_with_opinion_graph(N, M)
    show_degree_distribution(G)
    # nx.draw(G, with_labels=True)
    # plt.show()
