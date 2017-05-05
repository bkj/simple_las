#!/usr/bin/env python

"""
  expt_a9a.py
  
  Simplified version of
    https://github.com/AutonlabCMU/ActiveSearch/blob/sibi/kernelAS3/python/expt_a9a.py
"""

from __future__ import division

import gzip
import cPickle
import numpy as np

from simple_las import SimpleLAS

# --
# Helpers

def load_a9a(fname='./data/a9a_scaled_dataset.npz'):
  a9a = np.load(open(fname))
  
  x_tra  = a9a['x_tra']
  x_test = a9a['x_test']
  y_tra  = a9a['y_tra']
  y_test = a9a['y_test']
  
  X = np.r_[x_tra, x_test]
  Y = np.r_[y_tra, y_test].squeeze()
  Y = np.where(Y == 1, 1, 0)
  
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

prev = 0.05
alpha = 1e-6
eta = 0.5
n_init = 1
n_iter = 200

# --
# IO

X0, Y0 = load_a9a()
X0 /= np.sqrt((X0 ** 2).sum(axis=0))

# --
# Change prevalance

if Y0.mean() < prev:
  X, Y = X0, Y0
else:
  X, Y = change_prev(X0, Y0, prev)

act_prev = Y.mean()

# --
# Run SimpleLAS

def run(X, Y):
  init_pt = np.random.choice(np.where(Y)[0], n_init, replace=False)
  init_labels = {p:1 for p in init_pt}
  
  simp = SimpleLAS(X, init_labels=init_labels, pi=act_prev, eta=eta, alpha=alpha)
  
  hits = [1]
  for i in xrange(n_iter):
    idx = simp.next_message
    simp.setLabel(idx, Y[idx])
    hits.append(Y[idx])
  
  return hits

hist = {}
for i in range(10):
  print i
  hist[len(hist)] = run(X, Y)

# --

# from rsub import *
from matplotlib import pyplot as plt

avg = np.vstack(hist.values()).cumsum(axis=1).mean(axis=0)

np.vstack(hist.values()).mean(axis=0).cumsum()

_ = [plt.plot(np.cumsum(h), c='blue', alpha=0.2) for h in hist.values()]
_ = plt.plot(avg, c='red')
_ = plt.plot(np.arange(len(hits)), c='grey')
_ = plt.xlabel('# iterations')
_ = plt.ylabel('# hits')
_ = plt.title('Adult')
# show_plot()
plt.show()



