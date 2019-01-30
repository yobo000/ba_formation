# -*- coding: utf-8 -*-
"""
    PA predict, use link prediction score
    http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    PA score还不知道是用来做什么的,之后可能会有用，于是先使用原始算法
"""
import networkx as nx
import numpy as np

import collections

# G = nx.barabasi_albert_graph(10, 3)
G = nx.erdos_renyi_graph(5, 0.5)
bb = nx.betweenness_centrality(G)
print(bb)
G1 = G
G2 = G
print(G.edges())
print(G.degree())


def score(G):
    """
    Preferential attachment score will be computed for
    each pair of nodes given in the iterable.
    The pairs must be given as 2-tuples (u, v)
    where u and v are nodes in the graph.
    If ebunch is None then all non-existent edges in
    the graph will be used.
    Returns:
        piter – An iterator of 3-tuples in the form (u, v, p)
        where (u, v) is a pair of nodes and p is
        their preferential attachment score.
    """
    preds = nx.preferential_attachment(G, list(G.edges))
    predicts = np.array([p for u, v, p in preds])
    total_pr = sum(predicts)
    predicts = np.true_divide(predicts, total_pr)
    return predicts


def origin(G):
    """
    Calculate the probability by using origin method.
    """
    degrees = [d for n, d in G.degree()]
    degree_sequence = sorted(degrees, reverse=True)
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    print(deg, cnt)
    deg = np.array(deg)
    cnt = np.array(cnt)
    # sum deg * cnt
    total = np.sum(np.multiply(deg, cnt))
    print(total)
    # predict per node
    predicts = np.true_divide(np.array(degrees), total)
    print(predicts)
    values = np.random.binomial(1, predicts)
    print(values)
    endpoint = np.nonzero(values)
    return [int(x) for x in endpoint]


print(origin(G2))
# print(score(G1))
