"""
  regression-test.py
  
  Compare performance of `SimpleLAS` to reference implementation from 
    https://github.com/AutonlabCMU/ActiveSearch/blob/sibi/kernelAS3/python/activeSearchInterface.py
"""

from __future__ import division

import os
import sys
import gzip
import cPickle
import numpy as np
import scipy as sp

import activeSearchInterface as ASI
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
# Initialize search 

vsimp = SimpleLAS(X, init_labels=init_labels, pi=act_prev, eta=0.5, alpha=0.1)

params = ASI.Parameters(pi=act_prev, sparse=False, eta=0.5, alpha=0.1)
ref = ASI.linearizedAS(params)
ref.initialize(X, init_labels=init_labels)

assert((vsimp.f == ref.f).mean() == 1)

# --
# Compare performance

simp_idx = vsimp.next_message
vsimp.setLabel(simp_idx, Y[simp_idx])

ref_idx = ref.getNextMessage()
ref.setLabelCurrent(Y[ref_idx])

assert(ref_idx == simp_idx)
np.abs(vsimp.f - ref.f).max()
np.abs(vsimp.IM - ref.IM).max()
