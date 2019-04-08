# -*- coding: utf-8 -*-
import collections
import matplotlib.pyplot as plt
from sklearn import linear_model
import networkx as nx
import time
import requests
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

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


def upload_file(bucket_name, access_token, project_id, filename):
    url = "https://www.googleapis.com/upload/storage/v1/b/" + bucket_name + "/o"
    params = {
        'uploadType': "media",
        'name': filename
    }
    data = open('./' + filename, 'rb').read()
    headers = {
        'Authorization': 'Bearer {}'.format(access_token),
        "Content-Type": "image/png"
    }
    r = requests.post(url, params=params,
                      headers=headers, data=data)
    r.raise_for_status()
    """db = get_db()
    doc_ref = db.collection('result').document(str(int(time.time())))
    doc_ref.set({
        'name': filename,
        'time': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    })
    """
    return r.json()

def firebase_init(project_id):
        # Use the application default credentials
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'projectId': project_id})
    return


def get_db():
    db = firestore.client()
    return db


def graph_opinion(graph):
    opinions = np.array(list(nx.get_node_attributes(graph, 'opinion').values()))
    return opinions


def save_two_opinion_distribution(graph1, graph2, size_num, control, threshold, param, opinion, link, reversing):
    link = bool(link)
    opinion = bool(opinion)
    reversing = bool(reversing)
    filename = str(size_num) + '-' \
               + str(control) + '-' + str(threshold) + '-' \
               + str(param) + '-' + str(opinion)+ '-'+str(link)+'-'+str(reversing)+ time.strftime("%d%m") + "o.png"
    fig, (ax1, ax2) = plt.subplots(2, 1)
    opinions1 = graph_opinion(graph1)
    opinions2 = graph_opinion(graph2)
    bins = [0.01 * n for n in range(100)]
    ax2.hist(opinions2, bins=bins)
    ax2.set_title('Link-cut: '+ str(link)+', reversing: '+str(reversing)+', opinion: '+str(opinion))
    if control == "opinion":
        opinion = not opinion
    elif control == "link":
        link = not link
    elif control == "reversing":
        reversing = not reversing
    else:
        pass
    ax1.hist(opinions1, bins=bins)
    ax1.set_title('Link-cut: '+ str(link)+', reversing: '+str(reversing)+', opinion: '+str(opinion))
    plt.tight_layout()
    plt.savefig(filename)
    return filename


def save_record(record1, record2):
    fig = plt.figure()
    length1 = len(record1)
    length2 = len(record2)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, length1 + 1), record1, 'r', range(1, length2+1), record2, 'b')
    filename = "record.png"
    plt.savefig(filename)
    return filename


def graph_loglog(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    deg = np.array(deg)
    cnt = np.array(cnt)
    connectivity = np.divide(cnt, nx.number_of_nodes(G))
    deg_log = deg  # np.log10(deg)
    deg_cnt = connectivity  # np.log10(connectivity)
    order = np.argsort(deg_log)
    deg_log_array = np.array(deg_log)[order]
    deg_cnt_array = np.array(deg_cnt)[order]
    return deg_log_array, deg_cnt_array


def save_two_degree_distribution(G1, G2, size_num, control, threshold, param, opinion, link, reversing):
    link = bool(link)
    opinion = bool(opinion)
    reversing = bool(reversing)
    filename = str(size_num) + '-' \
               + str(control) + '-' + str(threshold) + '-' \
               + str(param) + '-' + str(opinion)+ '-'+str(link)+'-'+str(reversing)+ time.strftime("%d%m") + "d.png"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    deg_log_array2, deg_cnt_array2 = graph_loglog(G2)
    regr2 = linear_model.LinearRegression()
    regr2.fit(np.log10(deg_log_array2[3:13, np.newaxis]), np.log10(deg_cnt_array2[3:13]))
    gamma2 = regr2.coef_[0]
    ax.loglog(deg_log_array2, deg_cnt_array2, ".", color="blue", label='Link-cut: '+str(link)+', reversing: '+str(reversing)+', opinion: '+str(opinion)+'\n'+r'$\gamma$ = {0:.2f}'.format(gamma2))
    if control == "opinion":
        opinion = not opinion
    elif control == "link":
        link = not link
    elif control == "reversing":
        reversing = not reversing
    else:
        pass
    deg_log_array1, deg_cnt_array1 = graph_loglog(G1)
    regr1 = linear_model.LinearRegression()
    regr1.fit(np.log10(deg_log_array1[3:13, np.newaxis]), np.log10(deg_cnt_array1[3:13]))
    gamma1 = regr1.coef_[0]
    ax.loglog(deg_log_array1, deg_cnt_array1, ".", color="red", label='Link-cut: '+ str(link)+', reversing: '+str(reversing)+', opinion: '+str(opinion)+'\n'+r'$\gamma$ = {0:.2f}'.format(gamma1))
    # ax.plot(x1, y1, ".", color='red')
    plt.legend(loc='lower left')
    # plt.show()
    plt.savefig(filename)
    return filename
