#!/bin/sh
sudo apt update
sudo apt install -y python3-pip
git clone https://github.com/yobo000/ba_formation.git
cd ba_formation
sudo python3 -m pip install -r requirements.txt