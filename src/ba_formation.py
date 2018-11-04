# -*- coding: utf-8 -*-
"""
    generate ab graph with data
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import random
from operator import itemgetter

from networkx.utils import py_random_state
from networkx.generators.classic import empty_graph

N = 10000  # 10000
# edge pramas
M = 3  # 3
# growth node number
M_0 = 50
# iter times
T = 10
# deffuant tolerant
TOLERANT = 0.3
PRECISION = 0.00001  # 临时使用, 由于random_sample只在(0,1)
CONVERGENCE_COUNTER = 7  # 每20次不需要formation的次数


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
    # plt.savefig('result.png')


def show_opinion_distribution(G):
    opinions = np.array(list(nx.get_node_attributes(G, 'opinion').values()))
    # hist_data = np.histogram(opinions.values())
    bins = [0.01 * n for n in range(100)]
    plt.hist(opinions, bins=bins)
    plt.title("Histogram of opinions")
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


def formation(node1, node2):
    """
    value = node1.value + node2.value
    """
    value1 = G.node[node1]['opinion']
    value2 = G.node[node2]['opinion']
    diff = abs(value1 - value2)
    if diff > TOLERANT or diff < PRECISION:
        return False
    else:
        value = (value1 + value2) / 2
        return value


def is_not_convergence(loop, counter):
    if loop == 20:
        if counter >= CONVERGENCE_COUNTER:
            return 1, 1, False
        else:
            return 1, 0, True
    else:
        loop += 1
        return loop, counter, True


def opinion_formation(G):
    """
    1. 随机取edges
    2. formation
    3. 重复直到收敛
    only once: 只做一次，在最终N节点时
    period: 没加入n个节点时，做
    per node: 没加入一个节点便做一次

    在容忍度内, 取均值，容忍度外 断开edge # 断开edge，并未提及
    """
    counter = 0
    loop = 1
    not_convergence = True
    while not_convergence:
        edge = random.choice(list(G.edges(data=False)))
        # random edges
        value = formation(*edge)
        if value:
            node1, node2 = edge
            G.node[node1]['opinion'] = value
            G.node[node2]['opinion'] = value
        else:
            # remove edge
            # G.remove_edge(*edge)
            counter += 1
        loop, counter, not_convergence = is_not_convergence(loop, counter)
    return G


if __name__ == '__main__':
    G = barabasi_albert_with_opinion_graph(N, M)
    # show_degree_distribution(G)
    print(nx.number_of_edges(G))
    G = opinion_formation(G)
    show_opinion_distribution(G)
    # nx.draw(G, with_labels=True)
    # plt.show()
