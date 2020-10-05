#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n slas_env python=3.7
conda activate slas_env

pip install -r requirements.txt

pip install -e .

# --
# Get data

mkdir -p data
wget http://deeplearning.net/data/mnist/mnist.pkl.gz \
    -O ./data/mnist.pkl.gz