#!/bin/sh
cd ba_formation/src/task
# export FLASK_ENV=development
pkill -f "celery worker"
pkill -f "python3 run.py"
nohup celery worker -A run.celery --loglevel=debug --concurrency=1 > output_task.txt 2>&1 &
nohup python3 run.py > output.txt 2>&1 &