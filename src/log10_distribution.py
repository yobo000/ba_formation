# -*- coding: utf-8 -*-
"""
    Connectivity Distribution - log10 √
"""

import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

M = 10000
m = 3

G = nx.barabasi_albert_graph(M, m)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
deg = np.array(deg)
cnt = np.array(cnt)
# degrees = nx.number_of_edges(G)

connectivity = np.divide(cnt, M)
deg_log = deg  # np.log10(deg)
deg_cnt = connectivity  # np.log10(connectivity)

order = np.argsort(deg_log)
deg_log_array = np.array(deg_log)[order]
deg_cnt_array = np.array(deg_cnt)[order]
plt.loglog(deg_log_array, deg_cnt_array, ".")
# plt.plot(deg_log_array, deg_cnt_array, ".")

plt.show()
