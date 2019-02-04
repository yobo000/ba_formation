# -*- coding: utf-8 -*-
"""
    generate ab graph with data
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import random
from networkx.utils import py_random_state
from networkx.generators.classic import empty_graph
import requests
import time

# edge pramas 3
M = 3  # 3


class DissNetowrk(object):
    """docstring for DissNetowrk"""
    def __init__(self, **kw):
        super(DissNetowrk, self).__init__()
        self.size_num = kw["size_num"]
        self.init_num = kw["init_num"]
        self.growth = M
        self.loop_num = kw["loop_num"]
        self.threshold = kw["threshold"]
        self.param = kw["param"]
        self.graph = None
        self.filename = str(self.size_num) + '-' \
            + str(self.loop_num) + '-' + str(self.threshold) + '-' \
            + str(self.param)

    def opinion_thershold(self, opinion):
        max_value = opinion + self.threshold
        min_value = opinion - self.threshold
        end = 1 if max_value > 1 else max_value
        start = 0 if min_value < 0 else min_value
        return start, end

    def opinion_filter(self, opinion, pa_nodes):
        start, end = self.opinion_thershold(opinion)
        targets = list(filter(
            lambda x: x[1]["opinion"] < end and x[1]["opinion"] > start,
            pa_nodes))
        return list(map(lambda x: x[0], targets))

    def _random_subset(self, pa_nodes, seq, m, rng):
        """ Return m unique elements from seq.

        This differs from random.sample which can return repeated
        elements if seq holds repeated elements.

        Note: rng is a random.Random or numpy.random.RandomState instance.
        : Keep It!!!
        """
        targets = set()
        while len(targets) < m:
            x = rng.choice(seq)
            if x in pa_nodes:
                targets.add(x)
            else:
                pass
        return targets

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
        ax1.loglog(deg_log_array[2:], deg_cnt_array[2:], ".")
        ax1.set_ylim(0, 5)

        opinions = np.array(list(nx.get_node_attributes(self.graph, 'opinion').values()))
        # hist_data = np.histogram(opinions.values())
        bins = [0.01 * n for n in range(100)]
        ax2.hist(opinions, bins=bins)
        self.filename += '-' + time.strftime("%d%m")
        plt.savefig(self.filename)

    @py_random_state(1)
    def barabasi_albert_with_opinion_graph(self, seed=None):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.
        """
        if self.growth < 1 or self.growth >= self.size_num:
            raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                                   " and m < n, m = %d, n = %d" % (self.growth, self.size_num))
        # Add m initial nodes (m0 in barabasi-speak)
        self.graph = empty_graph(self.init_num)
        s = np.random.random_sample(self.init_num)
        # print("initial value:", dict(enumerate(s)))
        nx.set_node_attributes(self.graph, dict(enumerate(s)), 'opinion')
        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = list(self.graph.nodes(data=False))
        # Start adding the other n-m nodes. The first node is m.
        source = self.growth
        while source < self.size_num:
            opinion_value = np.random.random_sample()
            # Filter nodes from the targets
            nodes = list(self.graph.nodes(data=True))
            # pa_nodes = itemgetter(*targets)(nodes)
            pa_nodes = self.opinion_filter(opinion_value, nodes)
            targets = self._random_subset(pa_nodes, repeated_nodes, self.growth, seed)
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
            # targets = _random_subset(repeated_nodes, m, seed)
            source += 1
        return self.graph

    def formation(self, node1, node2):
        """
        value = node1.value + node2.value
        """
        value1 = self.graph.node[node1]['opinion']
        value2 = self.graph.node[node2]['opinion']
        diff = abs(value1 - value2)
        if diff < self.threshold and diff > PRECISION:
            value_1 = value1 - self.prama * (value1 - value2)
            value_2 = value2 - self.param * (value2 - value1)
            return value_1, value_2
        elif diff < PRECISION:
            return True, False
        else:
            return False, False

    def opinion_formation(self):
        """
        1. 随机取edges
        2. formation
        3. 重复直到收敛
        only once: 只做一次，在最终N节点时
        不过这里可能是个死循环
        """
        loop = 1
        # counter = 0
        # not_convergence = True
        while loop < self.loop_num:
            # G.edges 返回一个有两个node编号的tuple
            edges = list(self.graph.edges(data=False))
            if edges:
                edge = random.choice(edges)
                # random edges
                value1, value2 = self.formation(*edge)
                if value2:
                    node1, node2 = edge
                    self.graph.node[node1]['opinion'] = value1
                    self.graph.node[node2]['opinion'] = value2
                elif value1:
                    pass
                else:
                    # remove edge
                    self.graph.remove_edge(*edge)
                loop += 1
            else:
                break
        return self.graph

    def upload_file(self, bucket_name, access_token):
        url = "https://www.googleapis.com/upload/storage/v1/b/" + bucket_name + "/o"
        params = {
            'uploadType': "media",
            'name': self.filename
        }
        data = open('.' + self.filename, 'rb').read()
        headers = {
            'Authorization': 'Bearer {}'.format(access_token),
            "Content-Type": "image/png"
        }
        r = requests.post(url, params=params,
                          headers=headers, data=data)
        r.raise_for_status()

        return r.json()
