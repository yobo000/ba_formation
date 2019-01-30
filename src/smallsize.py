# -*- coding: utf-8 -*-
"""
    generate ab graph with data
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import random
import logging
from networkx.utils import py_random_state
from networkx.generators.classic import empty_graph
import argparse
import requests

N = 3000  # 10000
# edge pramas 3
M = 3  # 3
# growth node number 50
M_0 = 30
# iter times 10
T = 1
# perferential attachment tolerant
THERSHOLD = 0.3
# deffuant tolerant
TOLERANT = 0.3
DEFFUANT_COEFF = 0.5
PRECISION = 0.00001  # 由于random_sample只在(0,1)
LOOP_NUM = 500000  # 循环次数, 需要足够大

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename="output.log")
logger = logging.getLogger(__name__)

METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
SERVICE_ACCOUNT = 'default'


def get_access_token():
    url = '{}instance/service-accounts/{}/token'.format(
        METADATA_URL, SERVICE_ACCOUNT)

    # Request an access token from the metadata server.
    r = requests.get(url, headers=METADATA_HEADERS)
    r.raise_for_status()

    # Extract the access token from the response.
    access_token = r.json()['access_token']

    return access_token


def list_buckets(project_id, access_token):
    url = 'https://www.googleapis.com/storage/v1/b'
    params = {
        'project': project_id
    }
    headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }

    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()

    return r.json()


def upload_file(bucket_name, access_token):
    url = "https://www.googleapis.com/upload/storage/v1/b/" + bucket_name + "/o"
    params = {
        'uploadType': "media",
        'name': "result1_rnd.png"
    }
    data = open('./result1_rnd.png', 'rb').read()
    headers = {
        'Authorization': 'Bearer {}'.format(access_token),
        "Content-Type": "image/png"
    }
    r = requests.post(url, params=params,
                      headers=headers, data=data)
    r.raise_for_status()

    return r.json()

def save_distribution(G):
    fig, (ax1, ax2) = plt.subplots(2, 1)

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
    ax1.loglog(deg_log_array[2:], deg_cnt_array[2:], ".")
    ax1.set_ylim(0, 5)

    opinions = np.array(list(nx.get_node_attributes(G, 'opinion').values()))
    # hist_data = np.histogram(opinions.values())
    bins = [0.01 * n for n in range(100)]
    ax2.hist(opinions, bins=bins)

    plt.savefig('result_rnd.png')


def opinion_thershold(opinion):
    max_value = opinion + THERSHOLD
    min_value = opinion - THERSHOLD
    end = 1 if max_value > 1 else max_value
    start = 0 if min_value < 0 else min_value
    return start, end


def opinion_filter(opinion, pa_nodes):
    start, end = opinion_thershold(opinion)
    targets = list(filter(
        lambda x: x[1]["opinion"] < end and x[1]["opinion"] > start, pa_nodes))
    return list(map(lambda x: x[0], targets))


def _random_subset(pa_nodes, seq, m, rng):
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


@py_random_state(2)
def barabasi_albert_with_opinion_graph(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(M_0)
    s = np.random.random_sample(M_0)
    # print("initial value:", dict(enumerate(s)))
    nx.set_node_attributes(G, dict(enumerate(s)), 'opinion')
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = list(G.nodes(data=False))
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        opinion_value = np.random.random_sample()
        # Filter nodes from the targets
        nodes = list(G.nodes(data=True))
        # pa_nodes = itemgetter(*targets)(nodes)
        pa_nodes = opinion_filter(opinion_value, nodes)
        targets = _random_subset(pa_nodes, repeated_nodes, m, seed)
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
        # targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G


def formation(node1, node2):
    """
    value = node1.value + node2.value
    """
    value1 = G.node[node1]['opinion']
    value2 = G.node[node2]['opinion']
    diff = abs(value1 - value2)
    if diff < TOLERANT and diff > PRECISION:
        value_1 = value1 - DEFFUANT_COEFF * (value1 - value2)
        value_2 = value2 - DEFFUANT_COEFF * (value2 - value1)
        return value_1, value_2
    elif diff < PRECISION:
        return True, False
    else:
        return False, False


def opinion_formation(G):
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
    while loop < LOOP_NUM:
        # G.edges 返回一个有两个node编号的tuple
        edges = list(G.edges(data=False))
        if edges:
            edge = random.choice(edges)
            # random edges
            value1, value2 = formation(*edge)
            if value2:
                node1, node2 = edge
                G.node[node1]['opinion'] = value1
                G.node[node2]['opinion'] = value2
            elif value1:
                pass
            else:
                # remove edge
                G.remove_edge(*edge)
            loop += 1
        else:
            break
    return G


def main(project_id):
    G = barabasi_albert_with_opinion_graph(N, M)
    G = opinion_formation(G)
    save_distribution(G)
    access_token = get_access_token()
    buckets = list_buckets(project_id, access_token)
    bucket_name = buckets["items"][0]["id"]
    response = upload_file(bucket_name, access_token)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID.')
    args = parser.parse_args()
    main(args.project_id)
