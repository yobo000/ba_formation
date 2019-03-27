# -*- coding: utf-8 -*-
from flask import Flask, request
from celery import Celery
# from tasks import function1
import logging
from utils import *
from model1 import DissNetowrk


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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename="output.log")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
firebase_init("disstask")


@celery.task
def function1(project_id="", func_id=1, size_num=N, init_num=M_0, loop_num=50000, threshold=THERSHOLD, param=DEFFUANT_COEFF, link=1, reversing=1, opinion=1):
    # do once
    network = DissNetowrk(
        func_id=func_id,
        size_num=size_num,
        init_num=init_num,
        loop_num=loop_num,
        threshold=threshold,
        param=param,
        opinion=bool(opinion),
        link=bool(link),
        reversing=bool(reversing))
    network.barabasi_albert_with_opinion_graph()
    network.opinion_formation()
    network.save_distribution()
    access_token = get_access_token()
    buckets = list_buckets(project_id, access_token)
    bucket_name = buckets["items"][0]["id"]
    response = network.upload_file(bucket_name, access_token, project_id)
    return response


@celery.task
def function2(project_id="", func_id=2, size_num=N, init_num=M_0, loop_num=50000, threshold=THERSHOLD, param=DEFFUANT_COEFF, link=1, reversing=1, opinion=1):
    # for each node is adding
    network = DissNetowrk(
        func_id=func_id,
        size_num=size_num,
        init_num=init_num,
        loop_num=loop_num,
        threshold=threshold,
        param=param,
        opinion=bool(opinion),
        link=bool(link),
        reversing=bool(reversing))
    network.barabasi_albert_with_opinion_graph_formation()
    network.save_distribution()
    access_token = get_access_token()
    buckets = list_buckets(project_id, access_token)
    bucket_name = buckets["items"][0]["id"]
    response = network.upload_file(bucket_name, access_token, project_id)
    return response


@celery.task
def function3(project_id="", func_id=2, size_num=N, init_num=M_0, loop_num=50000, threshold=THERSHOLD, param=DEFFUANT_COEFF, link=1, reversing=1, opinion=1, control=""):
    network1 = DissNetowrk(
        func_id=func_id,
        size_num=size_num,
        init_num=init_num,
        loop_num=loop_num,
        threshold=threshold,
        param=param,
        opinion=bool(opinion),
        link=bool(link),
        reversing=bool(reversing))
    if control == "opinion":
        opinion = not opinion
    elif control == "link":
        link = not link
    elif control == "reversing":
        reversing = not reversing
    else:
        pass
    network2 = DissNetowrk(
        func_id=func_id,
        size_num=size_num,
        init_num=init_num,
        loop_num=loop_num,
        threshold=threshold,
        param=param,
        opinion=bool(opinion),
        link=bool(link),
        reversing=bool(reversing))
    graph1= network1.barabasi_albert_with_opinion_graph_formation()
    graph2 = network2.barabasi_albert_with_opinion_graph_formation()
    filename_d = save_two_degree_distribution(graph1, graph2, size_num, control, threshold, param, opinion, link, reversing)
    filename_o = save_two_opinion_distribution(graph1, graph2, size_num, control, threshold, param, opinion, link, reversing)
    access_token = get_access_token()
    buckets = list_buckets(project_id, access_token)
    bucket_name = buckets["items"][0]["id"]
    upload_file(bucket_name, access_token, project_id, filename_d)
    response = upload_file(bucket_name, access_token, project_id, filename_o)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return 'flask'
    elif request.method == 'POST':
        func_id = int(request.args.get('func'))
        project_id = request.args.get('id')
        size_num = int(request.args.get('size'))
        init_num = int(request.args.get('init'))
        loop_num = int(request.args.get('loop'))
        link_cut = int(request.args.get('link'))
        opinion = int(request.args.get('opinion'))
        reversing = int(request.args.get('reversing'))
        threshold = float(request.args.get('threshold'))
        param = float(request.args.get('param'))
        control = str(request.args.get('control', ''))
        if func_id == 1:
            task = function1.apply_async(kwargs={
                "project_id": project_id,
                "func_id": func_id,
                "size_num": size_num,
                "init_num": init_num,
                "loop_num": loop_num,
                "link": link_cut,
                "reversing": reversing,
                "threshold": threshold,
                "opinion": opinion,
                "param": param})
            return task.task_id
        elif func_id == 2:
            task = function2.apply_async(kwargs={
                "project_id": project_id,
                "func_id": func_id,
                "size_num": size_num,
                "init_num": init_num,
                "loop_num": loop_num,
                "link": link_cut,
                "reversing": reversing,
                "opinion": opinion,
                "threshold": threshold,
                "param": param})
            return task.task_id
        elif func_id == 3:
            task = function3.apply_async(kwargs={
                "project_id": project_id,
                "func_id": func_id,
                "size_num": size_num,
                "init_num": init_num,
                "loop_num": loop_num,
                "link": link_cut,
                "reversing": reversing,
                "opinion": opinion,
                "threshold": threshold,
                "param": param,
                "control": control})
            return task.task_id
        else:
            return "null"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
