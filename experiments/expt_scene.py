#!/usr/bin/env python

"""
  expt_mnist.py
  
  Simplified version of
    https://github.com/AutonlabCMU/ActiveSearch/blob/sibi/kernelAS3/python/expt_mnist.py
"""

from __future__ import division

import sys
import gzip
import cPickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.append('..')
from simple_las import SimpleLAS

# --
# Helpers

def change_prev(X, Y, prev):
  pos = np.where(Y != 0)[0]
  neg = np.where(Y == 0)[0]
  
  n = int(prev * neg.shape[0] / (1 - prev))
  pos = np.random.choice(pos, n, replace=False)
  
  inds = np.random.permutation(np.hstack([pos, neg]))
  return X[:,inds], Y[inds]

# --
# Params

prev = 0.05
n_init = 1
n_iter = 200

# --
# IO

df = pd.read_csv('./data/scene/db-crow.tsv', header=None, sep='\t')
labs = np.array(df[0])
vecs = np.array(df[df.columns[1:]])

labs = labs[:-1]
vecs = vecs[:-1]

X0 = normalize(vecs).T
Y0 = np.array([lab.split('/')[2] == 'pos' for lab in labs])

# --
# Change prevalance

if Y0.mean() < prev:
  X, Y = X0, Y0
else:
  X, Y = change_prev(X0, Y0, prev)

act_prev = Y.mean()

# --
# Seed labels

def run(X, Y):
  init_pt = np.random.choice(np.where(Y)[0], n_init, replace=False)
  init_labels = {p:1 for p in init_pt}
  
  simp = SimpleLAS(X, init_labels=init_labels, pi=act_prev, eta=0.5, alpha=1e-4)
  
  hits = [True]
  for i in xrange(n_iter):
    idx = simp.next_message
    simp.setLabel(idx, Y[idx])
    hits.append(Y[idx])
  
  return hits

hist = {}
for i in range(10):
  hist[len(hist)] = run(X, Y)

# --

from rsub import *
from matplotlib import pyplot as plt

avg = np.vstack(hist.values()).cumsum(axis=1).mean(axis=0)

_ = [plt.plot(np.cumsum(h), c='blue', alpha=0.2) for h in hist.values()]
_ = plt.plot(avg, c='red', alpha=0.5)
_ = plt.plot(np.arange(n_iter), c='grey', alpha=0.5)
_ = plt.plot(np.arange(n_iter) * act_prev, c='grey', alpha=0.5)
_ = plt.xlabel('# iterations')
_ = plt.ylabel('# hits')
_ = plt.title('Scene')
show_plot()

