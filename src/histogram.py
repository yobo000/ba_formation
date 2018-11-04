# -*- coding: utf-8 -*-
"""
    Degree Distribution
    BA model degree distribution p(k)=2*m^2/K^3
"""

import collections
import matplotlib.pyplot as plt
import networkx as nx
from numpy import true_divide, arange, log10

M = 3
G = nx.barabasi_albert_graph(1000, M)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
deg_sum = len(degree_sequence)
dist = true_divide(cnt, 1000)
cnt_log = list(map(log10, cnt))
dist_cnt = list(map(log10, dist))
plt.title("Degree Distribution")
plt.ylabel("Degree rate")
plt.xlabel("Degree")
plt.plot(cnt, dist, '*')
degree = arange(min(cnt), max(cnt), 1)
numerator = 2 * M * M
plt.plot(degree, numerator / pow(degree, 3), "-")
plt.show()
