#!/usr/bin/env python

"""
    nwpu.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn import metrics

from rsub import *
from matplotlib import pyplot as plt

from simple_las import SimpleLAS, RandomSearch

np.random.seed(123)

# --
# Params

eta        = 0.5 # !! What does this do?
alpha      = 0   # !! What does this do?
n_init     = 1   # Number of initial labels
n_iter     = 200 # Number of iterations

# --
# IO

base_dir = '/home/bjohnson/projects/rs_query/runs/'
X        = np.load(os.path.join(base_dir, 'NWPU-RESISC45/X.npy'))
labs     = np.load(os.path.join(base_dir, 'NWPU-RESISC45/y.npy'))
paths    = np.load(os.path.join(base_dir, 'NWPU-RESISC45/paths.npy'))

X = X.astype(np.float64) # !! Big speedup

# p = np.random.permutation(X.shape[0])
# X, labs, paths = X[p], labs[p], paths[p]

# --
# Postprocess features

X = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))

# !! Whitening / centering breaks this badly -- why?

# --
# Run SimpleLAS

target_class = np.unique(labs)[40]
y            = labs == target_class
# >>
# init_labels  = [10] # np.random.choice(np.where(y)[0], n_init, replace=False)
# --
np.random.seed(345)

n_pos   = (y == 1).sum()
sel_pos = np.random.choice(np.where(y == 1)[0], int(n_pos * 0.1), replace=False)
sel_neg = np.where(y == 0)[0]
sel     = np.hstack([sel_pos, sel_neg])
# sel     = np.random.permutation(sel)

X, y = X[sel], y[sel]

init_labels = [np.where(y == 1)[0][10]]
init_vec    = X[np.array(init_labels)]
# <<
init_labels = {p:1 for p in init_labels}

# >>
scores = (X @ init_vec.T).squeeze()
null   = np.cumsum(y[np.argsort(-scores)])
# <<


samplers = {
  "las"  : SimpleLAS(X, init_labels=init_labels, pi=y.mean(), eta=eta, alpha=alpha),
  "mean" : RandomSearch(X, init_labels=init_labels, query_mean=True),
  # "min"  : RandomSearch(X, init_labels=init_labels, query_mean=False),
}

n_iter = 100



hist = []
for sampler_name,sampler in samplers.items():
  for i in range(n_iter):
    idx = sampler.get_next_message()
    sampler.set_label(idx, y[idx])
    
    uidxs = np.array(sampler.unlabeled_idxs)
    uy    = y[uidxs]
    uf    = sampler.f[uidxs]
    
    hist.append({
      "sampler_name" : sampler_name,
      "iter"         : i,
      "idx"          : int(idx),
      "lab"          : y[int(idx)], 
      "roc"          : metrics.roc_auc_score(uy, uf),
      "p010"         : uy[np.argsort(-uf)[:10]].mean(),
      "p050"         : uy[np.argsort(-uf)[:50]].mean(),
      "p100"         : uy[np.argsort(-uf)[:100]].mean(),
      "p250"         : uy[np.argsort(-uf)[:250]].mean(),
      "p500"         : uy[np.argsort(-uf)[:500]].mean(),
    })
    print(hist[-1])

hist = pd.DataFrame(hist)

# --
# Plot

for sampler_name in hist.sampler_name.unique():
  sub = hist[hist.sampler_name == sampler_name]
  # _ = plt.plot(sub.iter, sub.p500, label=sampler_name, alpha=0.75)
  _ = plt.plot(sub.iter, sub.roc, label=sampler_name, alpha=0.75)

_ = plt.legend()
_ = plt.grid('both', alpha=0.25)
show_plot()


for sampler_name in hist.sampler_name.unique():
  sub = hist[hist.sampler_name == sampler_name]
  _ = plt.plot(sub.iter, sub.lab.cumsum(), label=sampler_name, alpha=0.75)

_ = plt.plot(null[:sub.iter.max()])
_ = plt.plot([0, y.sum()], [0, y.sum()], c='grey', alpha=0.25)
_ = plt.legend()
_ = plt.grid('both', alpha=0.25)
show_plot()

