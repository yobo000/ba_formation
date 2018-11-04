# -*- coding: utf-8 -*-
"""
    connectivity distibution
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# scale
N = 10
# edge
M = 3
# growth node number
M_0 = 5
# iter times
T = 10

G = nx.barabasi_albert_graph(N, M)
nx.draw(G, with_labels=True)


def calc_pa_predict(G):
    pred = nx.preferential_attachment(G, list(G.edges))
    predicts = np.array(list(map(lambda x: x[-1], pred)))
    MAX_PR = max(predicts)
    predicts = np.array([pr / MAX_PR for pr in predicts])
    print(predicts)
    # convert to 2-d array(multi-proc), if not, 
    # raise warnning and low efficiency
    values = np.random.binomial(1, predicts)
    return values


def add_node(G):
    """
        1. new a node
        2. calculate predicts
        3. choose edge node
        4. connect node
        5. show connectivity & count edge
    """
    G.add_node(M_0)
    connectivity = nx.edge_connectivity(G)
    print(connectivity)


print(calc_pa_predict(G))
