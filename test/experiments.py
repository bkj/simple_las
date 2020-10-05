#!/usr/bin/env python

"""
  experiments.py
  
  Simplified version of experiments in 
  
    https://github.com/AutonlabCMU/ActiveSearch/blob/sibi/kernelAS3/python
  
  Note only the MNIST dataset is publicly availabe ATM
"""

import sys
import numpy as np
import _pickle as pickle
from tqdm import trange

from rsub import *
from matplotlib import pyplot as plt

from simple_las import SimpleLAS

# --
# Helpers

def load_mnist(fname='./data/mnist.pkl'):
  train_set, valid_set, test_set = pickle.load(open(fname, 'rb'), encoding='latin1')
  X = np.r_[train_set[0], valid_set[0], test_set[0]]
  y = np.r_[train_set[1], valid_set[1], test_set[1]]
  return X, y

def change_prevalence(X, y, prev):
  pos = np.where(y != 0)[0]
  neg = np.where(y == 0)[0]
  
  n   = int(prev * neg.shape[0] / (1 - prev))
  pos = np.random.choice(pos, n, replace=False)
  
  inds = np.random.permutation(np.hstack([pos, neg]))
  return X[inds], y[inds]

# --
# Params

prevalence = 0.05
eta        = 0.5
n_init     = 1
n_iter     = 200

# --
# IO

X0, y0       = load_mnist()
alpha        = 0
target_class = 1
y0           = y0 == target_class

X0 /= np.sqrt((X0 ** 2).sum(axis=1, keepdims=True))

# --
# Change prevalance

if y0.mean() < prevalence:
  X, y = X0, y0
else:
  X, y = change_prevalence(X0, y0, prevalence)

act_prevalence = y.mean()

# --
# Run SimpleLAS

def run_activesearch(X, y):
  init_pt     = np.random.choice(np.where(y)[0], n_init, replace=False)
  init_labels = {p:1 for p in init_pt}
  
  print('SimpleLAS: __init__', file=sys.stderr)
  las = SimpleLAS(X, init_labels=init_labels, pi=act_prevalence, eta=eta, alpha=alpha)
  
  print('SimpleLAS: run', file=sys.stderr)
  hits = [1]
  for i in trange(n_iter):
    idx = las.next_message
    print(idx, y[idx])
    las.set_label(idx, y[idx])
    hits.append(y[idx])
  
  return hits

n_experiments = 1

results = {}
for i in range(n_experiments):
  results[i] = run_activesearch(X, y)

# --
# Plot performance

avg = np.vstack(results.values()).cumsum(axis=1).mean(axis=0)
_ = [plt.plot(np.cumsum(h), c='blue', alpha=0.2) for h in hist.values()]
_ = plt.plot(avg, c='red', alpha=0.5)
_ = plt.plot(np.arange(n_iter), c='grey', alpha=0.5)
_ = plt.plot(np.arange(n_iter) * act_prevalence, c='grey', alpha=0.5)
_ = plt.xlabel('# iterations')
_ = plt.ylabel('# hits')
_ = plt.title('mnist active search')
show_plot()

