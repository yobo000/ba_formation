# -*- coding: utf-8 -*-
"""
    set oponion
    TODO: initial data format
"""
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
s = np.random.poisson(5, N)
# set_node_attributes second params should be dict-like
nx.set_node_attributes(G, dict(enumerate(s)), 'opinion')
print("Set value for node:")
print(list(G.nodes(data=True)))
print("Nodes number: %d " % G.number_of_nodes())
# add n for loop
# G.add_nodes_from(range(N + 1, N + M_0), attr_dict={"it": 1, "opinion": 0.5})
G.add_nodes_from(range(N + 1, N + M_0), opinion=0.5)
print("After add nodes number: %d " % G.number_of_nodes())
print(list(G.nodes(data=True)))
node = G.node[2]
print(node)
nodes = list(G.nodes(data=True))
targets = list(filter(lambda x: x[1]["opinion"] < 4 and x[1]["opinion"] > 1, nodes))
print(list(map(lambda x: x[0], targets)))
