#!/usr/bin/env python

"""
  experiments.py
  
  Simplified version of experiments in 
  
    https://github.com/AutonlabCMU/ActiveSearch/blob/sibi/kernelAS3/python
  
  Note only the MNIST dataset is publicly availabe ATM
"""

from __future__ import division

import sys
import gzip
import cPickle
import numpy as np

from matplotlib import pyplot as plt

sys.path.append('..')
from simple_las import SimpleLAS

# --
# Helpers

def load_mnist(fname='./data/mnist.pkl.gz'):
  train_set, valid_set, test_set = cPickle.load(gzip.open(fname, 'rb'))
  X = np.r_[train_set[0], valid_set[0], test_set[0]]
  y = np.r_[train_set[1], valid_set[1], test_set[1]]
  return X, y


def load_a9a(fname='./data/a9a_scaled_dataset.npz', cl=1):
  a9a = np.load(open(fname))
  
  x_tra  = a9a['x_tra']
  x_test = a9a['x_test']
  y_tra  = a9a['y_tra']
  y_test = a9a['y_test']
  
  X = np.r_[x_tra, x_test]
  y = np.r_[y_tra, y_test].squeeze()
  y = np.where(y == 1, 1, 0)
  
  return X, y == cl


def change_prev(X, y, prev):
  pos = np.where(y != 0)[0]
  neg = np.where(y == 0)[0]
  
  n = int(prev * neg.shape[0] / (1 - prev))
  pos = np.random.choice(pos, n, replace=False)
  
  inds = np.random.permutation(np.hstack([pos, neg]))
  return X[inds], y[inds]

# --
# Params

dataset = 'mnist'
prev = 0.05
eta = 0.5
n_init = 1
n_iter = 200

# --
# IO

if dataset == 'mnist':
  X0, y0 = load_mnist()
  alpha = 0
  cl = 1
  y0 = y0 == cl
elif dataset == 'a9a':
  X0, y0 = load_a9a()
  alpha = 1e-6
else:
  raise Exception()

X0 /= np.sqrt((X0 ** 2).sum(axis=1, keepdims=True))

# --
# Change prevalance

if y0.mean() < prev:
  X, y = X0, y0
else:
  X, y = change_prev(X0, y0, prev)

act_prev = y.mean()

# --
# Run SimpleLAS

def run_activesearch(X, y):
  init_pt = np.random.choice(np.where(y)[0], n_init, replace=False)
  init_labels = {p:1 for p in init_pt}
  
  simp = SimpleLAS(X, init_labels=init_labels, pi=act_prev, eta=eta, alpha=alpha)
  
  hits = [1]
  for i in xrange(n_iter):
    idx = simp.next_message
    simp.setLabel(idx, y[idx])
    hits.append(y[idx])
  
  return hits

n_experiments = 10
results = {}
for i in range(n_experiments):
  results[i] = run_activesearch(X, y)

# --
# Plot performance

avg = np.vstack(hist.values()).cumsum(axis=1).mean(axis=0)
_ = [plt.plot(np.cumsum(h), c='blue', alpha=0.2) for h in hist.values()]
_ = plt.plot(avg, c='red', alpha=0.5)
_ = plt.plot(np.arange(n_iter), c='grey', alpha=0.5)
_ = plt.plot(np.arange(n_iter) * act_prev, c='grey', alpha=0.5)
_ = plt.xlabel('# iterations')
_ = plt.ylabel('# hits')
_ = plt.title(dataset)
show_plot()

