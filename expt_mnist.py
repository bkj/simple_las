"""
  expt_mnist-v2.py
  
  Cleaned up + simplified version of `expt_mnist`
"""

from __future__ import division

import gzip
import cPickle
import numpy as np

from simple_las import SimpleLAS

# --
# Helpers

def load_mnist(fname='./mnist.pkl.gz'):
  train_set, valid_set, test_set = cPickle.load(gzip.open(fname, 'rb'))
  X = np.r_[train_set[0], valid_set[0], test_set[0]]
  Y = np.r_[train_set[1], valid_set[1], test_set[1]]
  return X.T, Y

def change_prev(X, Y, prev):
  pos = np.where(Y != 0)[0]
  neg = np.where(Y == 0)[0]
  
  n = int(prev * neg.shape[0] / (1 - prev))
  pos = np.random.choice(pos, n, replace=False)
  
  inds = np.random.permutation(np.hstack([pos, neg]))
  return X[:,inds], Y[inds]

# --
# Params

cl = 1
prev = 0.05
n_init = 1
n_iter = 400

# --
# IO

X0, labs = load_mnist()
X0 /= np.sqrt((X0 ** 2).sum(axis=0))
Y0 = labs == cl

# --
# Change prevalance

if Y0.mean() < prev:
  X, Y = X0, Y0
else:
  X, Y = change_prev(X0, Y0, prev)

act_prev = Y.mean()

# --
# Seed labels

init_pt = np.random.choice(np.where(Y)[0], n_init, replace=False)
init_labels = {p:1 for p in init_pt}

# --
# Run SimpleLAS

simp = SimpleLAS(X, init_labels=init_labels, pi=act_prev, eta=0.5, alpha=0)

hits = [True]
for i in xrange(n_iter):
  idx = simp.next_message
  simp.setLabel(idx, Y[idx])
  hits.append(Y[idx])

