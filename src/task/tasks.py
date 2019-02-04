# -*- coding: utf-8 -*-
"""
    generate ab graph with data
"""
import celery
from .utils import get_access_token, list_buckets
from .model1 import DissNetowrk


N = 3000  # 10000
# edge pramas 3
M = 3  # 3
# growth node number 50
M_0 = 30
# iter times 10
T = 1
# perferential attachment tolerant
THERSHOLD = TOLERANT = 0.3
# deffuant tolerant
DEFFUANT_COEFF = 0.5
PRECISION = 0.00001  # 由于random_sample只在(0,1)


@celery.task
def function1(project_id="", size_num=N, init_num=M_0, loop_num=50000, threshold=THERSHOLD, param=DEFFUANT_COEFF):
    network = DissNetowrk(
        size_num=size_num,
        init_num=init_num,
        loop_num=loop_num,
        threshold=threshold,
        param=param)
    network.barabasi_albert_with_opinion_graph()
    network.opinion_formation()
    network.save_distribution()
    access_token = get_access_token()
    buckets = list_buckets(project_id, access_token)
    bucket_name = buckets["items"][0]["id"]
    response = network.upload_file(bucket_name, access_token)
    return response
