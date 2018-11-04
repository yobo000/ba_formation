# -*- coding: utf-8 -*-
"""
    connectivity distibution
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections

# deffuent and growth at the same time
# scale
N = 10000
# edge pramas
M = 3
# growth node number
M_0 = 50
# iter times
T = 10


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


def calc_pa_predict(G):
    preds = nx.preferential_attachment(G, list(G.edges))
    predicts = np.array([p for u, v, p in preds])
    total_pr = sum(predicts)
    predicts = np.true_divide(predicts, total_pr)
    """
        convert to 2-d array(multi-proc), if not,
        raise warnning and low efficiency
    """
    values = np.random.binomial(1, predicts)
    return values


def calc_pa_predict2(G):
    degrees = [d for n, d in G.degree()]
    degree_sequence = sorted(degrees, reverse=True)
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    deg = np.array(deg)
    cnt = np.array(cnt)
    # sum deg * cnt
    total = np.sum(np.multiply(deg, cnt))
    # predict per node
    predicts = np.true_divide(np.array(degrees), total)
    values = np.random.binomial(1, predicts)
    return values


def add_node(it, G):
    """
        1. new a node
        2. calculate predicts
        3. choose edge node
        4. connect node
        5. show connectivity & count edge
    """
    print("iteration times:", it)
    start = N + it * M_0 + 1
    end = N + M_0 + it * M_0
    print("add node from: %d -> %d" % (start, end))
    new_node = range(start, end)
    G.add_nodes_from(new_node)
    for n in new_node:
        link_result = calc_pa_predict2(G)
        endpoints = np.nonzero(link_result)[0]
        new_edges = [{n, endpont} for endpont in endpoints]
    G.add_edges_from(new_edges)


def main():
    G = nx.barabasi_albert_graph(N, M)
    # print("Initial Connectivity:", nx.edge_connectivity(G))
    for i in range(T):
        add_node(i, G)
    show_degree_distribution(G)


if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    period = end - start
    print("Cose: {0} seconds".format(period.seconds))
