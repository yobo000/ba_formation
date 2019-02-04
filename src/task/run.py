# -*- coding: utf-8 -*-
from flask import Flask, request
from celery import Celery
from tasks import function1
import logging

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


@app.route('/', methods=['POST'])
def longtask():
    func_id = request.args.get('func')
    project_id = request.args.get('id')
    size_num = request.args.get('size')
    init_num = request.args.get('init')
    loop_num = request.args.get('loop')
    threshold = request.args.get('threshold')
    param = request.args.get('prama')
    if func_id == 1:
        task = function1.apply_async(
            project_id=project_id,
            size_num=size_num,
            init_num=init_num,
            loop_num=loop_num,
            threshold=threshold,
            param=param)
        return {"result": task.id}
    else:
        return {"result": "null"}
