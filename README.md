## Opinion formation on the Barabasi-Albert Model
The purpose is for my dissertation project.

![Python >= 3.5 ](https://img.shields.io/badge/python-%3E%3D3.5-blue.svg)


#### Usage
I mainly used [NetowrkX](https://networkx.github.io/) and Matplotlib.
The `.py` files in the **src** directory shows some examples.
```
# use pipenv or virtualenv[optional]
pip install -r requirements.txt
python src/*.py
```

#### For GCP

The production files store in the **src/task**. You can use HTTP post to call a task.

Requirement:
  * Redis
  * celery
  * Flask
  * google-cloudestorage-api

You can install the requirements.txt in **src/task** using `pip`.
Then use `src/task/start.sh` or `src/task/restart.sh`
