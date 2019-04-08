# -*- coding: utf-8 -*-
"""
    generate ab graph with data
"""
import matplotlib.pyplot as plt
import networkx as nx
from operator import itemgetter
import numpy as np
from functools import reduce
from sklearn import linear_model
import collections
import random
from networkx.utils import py_random_state
from networkx.generators.classic import complete_graph, empty_graph
import requests
import time

from utils import *

# edge pramas 3
M = 3  # 3
PRECISION = 0.00001 # 由于random_sample只在(0,1)
INIT_SIZE = 300


class DissNetowrk(object):
    """docstring for DissNetowrk"""
    def __init__(self, **kw):
        super(DissNetowrk, self).__init__()
        self.size_num = kw["size_num"]
        self.init_num = kw["init_num"]
        self.func_id = kw["func_id"]
        self.growth = M
        self.loop_num = kw["loop_num"]
        self.threshold = kw["threshold"]
        self.param = kw["param"]
        self.graph = None
        self.opinion = kw['opinion']
        self.link_cut = kw["link"]
        self.reversing = kw["reversing"]
        self.degree_record =[]
        self.d = 0
        if self.func_id == 1:
            self.filename = str(self.size_num) + '-' \
                + str(self.loop_num) + '-' + str(self.threshold) + '-' \
                + str(self.param)
        elif self.func_id == 2:
            self.filename = str(self.size_num) + '-' \
                + 'inGrowth' + '-' + str(self.threshold) + '-' \
                + str(self.param)
        else:
            pass

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
        target_nodes = {}
        for node in targets:
            target_nodes[node[0]] = True
        # return list(map(lambda x: x[0], targets))
        return target_nodes

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
            # if x in pa_nodes:
            if pa_nodes.get(x, False):
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
        connectivity = np.divide(cnt, self.size_num)
        deg_log = deg  # np.log10(deg)
        deg_cnt = connectivity  # np.log10(connectivity)
        order = np.argsort(deg_log)
        deg_log_array = np.array(deg_log)[order]
        deg_cnt_array = np.array(deg_cnt)[order]
        x, y = map(np.log, [deg_log_array[2:], deg_cnt_array[2:]])

        cut = int(self.size_num / 6)
        ax1.plot(x, y, ".")
        regr = linear_model.LinearRegression()
        regr.fit(x[:cut, np.newaxis], y[:cut])
        ax1.loglog(deg_log_array[2:], deg_cnt_array[2:], ".")
        ax1.plot(x, regr.predict(x[:, np.newaxis]), color='red', linewidth=3)

        opinions = np.array(list(nx.get_node_attributes(self.graph, 'opinion').values()))
        # hist_data = np.histogram(opinions.values())
        bins = [0.01 * n for n in range(100)]
        ax2.hist(opinions, bins=bins)
        self.filename += '-' + time.strftime("%d%m") + ".png"
        plt.savefig(self.filename)

    def get_degree_evolution(self):
        return self.degree_record

    def save_degree_opinion_distribution(self, count):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        nodes = nx.get_node_attributes(self.graph, 'opinion')
        opinions = []
        degrees = []
        for node, value in nodes.items():
            opinions.append(value)
            degrees.append(len(nx.edges(self.graph, node)))
        nopinions = np.array(opinions)
        ndegrees = list(map(np.log, np.array(degrees)))
        ax.plot(nopinions, ndegrees, ".", color='blue')
        filename = 'dots-'+str(count)+ str(self.link_cut) + str(self.reversing) + str(self.opinion) +'-'+ time.strftime("%d%m") + ".png"
        plt.savefig(filename)
        return filename

    def time_step_distribution(self, ax=None):
        if ax:
            return ax
        else:
            return ax

    def save_inter_distribution(self, count):
        """
        1. print degree-opinion dots
        2. degree distribution separate by opinion
        """
        thresholds = (0.3,0.7)
        titles= ["0-0.3", "0.3-0.7", "0.7-1"]
        geq = []
        leq = []
        mid = []
        geq_degree_sequence = []
        leq_degree_sequence = []
        mid_degree_sequence = []
        # degree_sequence = []
        fig, axes = plt.subplots(nrows=2, ncols=3)
        start = 0
        nodes = nx.get_node_attributes(self.graph, 'opinion')
        for node, value in nodes.items():
            if value >= thresholds[-1]:
                geq.append(node)
                geq_degree_sequence.append(len(nx.edges(self.graph, node)))
            elif value <= thresholds[0]:
                leq.append(node)
                leq_degree_sequence.append(len(nx.edges(self.graph, node)))
            else:
                mid.append(node)
                mid_degree_sequence.append(len(nx.edges(self.graph, node)))
        degree_sequence = [leq_degree_sequence, mid_degree_sequence, geq_degree_sequence]
        filter_nodes = [leq, mid, geq]
        for i in range(3):
            degreeCount = collections.Counter(degree_sequence[i])
            deg, cnt = zip(*degreeCount.items())
            deg = np.array(deg)
            cnt = np.array(cnt)
            connectivity = np.divide(cnt, self.size_num)
            deg_log = deg  # np.log10(deg)
            deg_cnt = connectivity  # np.log10(connectivity)
            order = np.argsort(deg_log)
            deg_log_array = np.array(deg_log)[order]
            deg_cnt_array = np.array(deg_cnt)[order]
            axes[0, i].loglog(deg_log_array, deg_cnt_array, ".")
            axes[0, i].set_title(titles[i])
            # axes[0, i].get_yaxis().set_visible(False)
            # axes[0, i].get_xaxis().set_visible(False)
            regr = linear_model.LinearRegression()
            if len(deg_log_array) > 12:
                regr.fit(np.log10(deg_log_array[3:13, np.newaxis]), np.log10(deg_cnt_array[3:13]))
                gamma = regr.coef_[0]
            else:
                gamma = 0.0
            axes[0, i].set_ylabel(r'$\gamma$ = {0:.2f}'.format(gamma))

            opinions = np.array(itemgetter(*filter_nodes[i])(list(nx.get_node_attributes(self.graph, 'opinion').values())))
            if i == 1:
                bins = [0.3 + 0.01 * n for n in range(70)]
                hist_range = (0.3, 0.7)
            elif i == 0:
                bins = [0.01 * n for n in range(30)]
                hist_range = (0, 0.3)
            else:
                bins = [0.7 + 0.01 * n for n in range(30)]
                hist_range = (0.7 ,1)
            axes[1, i].hist(opinions, bins=bins)# , range=hist_range)
        filename = str(count) + str(self.link_cut) + str(self.reversing) + str(self.opinion) +'-' + time.strftime("%d%m") + ".png"
        plt.tight_layout()
        plt.savefig(filename)
        return filename

    @py_random_state(1)
    def barabasi_albert_with_opinion_graph(self, seed=None):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.
        """
        if self.growth < 1 or self.growth >= self.size_num:
            raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                                   " and m < n, m = %d, n = %d" % (self.growth, self.size_num))
        # Add m initial nodes (m0 in barabasi-speak)
        self.graph = complete_graph(self.init_num)
        s = np.random.random_sample(self.init_num)
        # print("initial value:", dict(enumerate(s)))
        nx.set_node_attributes(self.graph, dict(enumerate(s)), 'opinion')
        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = list(self.graph.nodes(data=False))
        # Start adding the other n-m nodes. The first node is m.
        source = self.growth
        while source < self.size_num:
            pa_nodes = {}
            opinion_value = np.random.random_sample()
            # Filter nodes from the targets
            nodes = list(self.graph.nodes(data=True))
            # pa_nodes = itemgetter(*targets)(nodes)
            if self.opinion:
                pa_nodes = self.opinion_filter(opinion_value, nodes)
            else:
                for node in nodes:
                    pa_nodes[node[0]] = True
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

    @py_random_state(1)
    def barabasi_albert_with_opinion_graph_formation(self, seed=None):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.
        """
        if self.growth < 1 or self.growth >= self.size_num:
            raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                                   " and m < n, m = %d, n = %d" % (self.growth, self.size_num))
        # Add m initial nodes (m0 in barabasi-speak)
        self.graph = complete_graph(self.init_num)
        s = np.random.random_sample(self.init_num)
        # print("initial value:", dict(enumerate(s)))
        nx.set_node_attributes(self.graph, dict(enumerate(s)), 'opinion')
        repeated_nodes = list(self.graph.nodes(data=False))
        # Start adding the other n-m nodes. The first node is m.
        source = self.growth
        while source < INIT_SIZE:
            opinion_value = np.random.random_sample()
            nodes = list(self.graph.nodes(data=True))
            pa_nodes = self.opinion_filter(opinion_value, nodes)
            targets = self._random_subset(pa_nodes, repeated_nodes, self.growth, seed)
            self.graph.add_node(source, opinion=opinion_value)
            self.graph.add_edges_from(zip([source] * self.growth, targets))
            repeated_nodes.extend(targets)
            repeated_nodes.extend([source] * self.growth)
            source += 1
        while source < self.size_num:
            opinion_value = np.random.random_sample()
            nodes = list(self.graph.nodes(data=True))
            # pa_nodes = self.opinion_filter(opinion_value, nodes)
            if self.opinion:
                pa_nodes = self.opinion_filter(opinion_value, nodes)
            else:
                for node in nodes:
                    pa_nodes[node[0]] = True
            targets = self._random_subset(pa_nodes, repeated_nodes, self.growth, seed)
            self.graph.add_node(source, opinion=opinion_value)
            self.graph.add_edges_from(zip([source] * self.growth, targets))
            # self.d += self.growth
            repeated_nodes.extend(targets)
            repeated_nodes.extend([source] * self.growth)
            # formation in growth
            self.opinion_formation_in_growth()
            source += 1
            if source in [500, 1000, 1500, 2000]:
                filename1 = self.save_degree_opinion_distribution(source)
                filename2 = self.save_inter_distribution(source)
                access_token = get_access_token()
                buckets = list_buckets('disstask', access_token)
                bucket_name = buckets["items"][0]["id"]
                upload_file(bucket_name, access_token, 'disstask', filename1)
                upload_file(bucket_name, access_token, 'disstask', filename2)
        return self.graph

    @py_random_state(2)
    def reversing_proc(self, node, seed=None):
        """
        for the link-cut node add a new link on it
        """
        nodes = list(self.graph.nodes(data=True))
        value = self.graph.node[node]['opinion']
        if self.opinion:
            pa_nodes = self.opinion_filter(value, nodes)
        else:
            pa_nodes = {}
            for node_i in nodes:
                pa_nodes[node_i[0]] = True
        repeated_nodes = list(reduce(lambda y1, y2: y1 + y2, map(lambda x: [x] * len(list(nx.neighbors(self.graph, x))), pa_nodes.keys())))
        targets = self._random_subset(pa_nodes, repeated_nodes, 1, seed)
        self.graph.add_edges_from([(node, targets.pop())])

    def formation(self, node1, node2):
        """
        value = node1.value + node2.value
        """
        value1 = self.graph.node[node1]['opinion']
        value2 = self.graph.node[node2]['opinion']
        diff = abs(value1 - value2)
        if diff < self.threshold and diff > PRECISION:
            value_1 = value1 - self.param * (value1 - value2)
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
                    node1, node2 = edge
                    # link-cut
                    if self.link_cut:
                        self.graph.remove_edge(*edge)
                        if self.reversing:
                            self.reversing_proc(node1)
                            self.reversing_proc(node2)
                loop += 1
            else:
                break
        return self.graph

    def opinion_formation_in_growth(self):
        """
        1. 随机取edges
        2. formation
        3. 重复直到收敛
        do 40 - 100 times for each node
        """
        loop = 1
        # counter = 0
        # not_convergence = True
        loop_num = np.random.randint(50, 100)
        while loop < loop_num:
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
                    node1, node2 = edge
                    # link-cut
                    if self.link_cut:
                        self.graph.remove_edge(*edge)
                        if self.reversing:
                            rnode = random.choice(edge)
                            self.reversing_proc(rnode)
                            # self.reversing_proc(node2)
                loop += 1
            else:
                break
        return self.graph

    def upload_file(self, bucket_name, access_token, project_id):
        url = "https://www.googleapis.com/upload/storage/v1/b/" + bucket_name + "/o"
        params = {
            'uploadType': "media",
            'name': self.filename
        }
        data = open('./' + self.filename, 'rb').read()
        headers = {
            'Authorization': 'Bearer {}'.format(access_token),
            "Content-Type": "image/png"
        }
        r = requests.post(url, params=params,
                          headers=headers, data=data)
        r.raise_for_status()
        db = get_db()
        doc_ref = db.collection('result').document(str(int(time.time())))
        doc_ref.set({
            'name': self.filename,
            'time': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        })
        return r.json()
