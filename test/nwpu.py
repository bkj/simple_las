
import os
import sys
import numpy as np
from tqdm import trange
from sklearn.decomposition import PCA

from rsub import *
from matplotlib import pyplot as plt

from simple_las import SimpleLAS

# --
# Params

eta        = 0.5
n_init     = 1
n_iter     = 200
alpha      = 0

# --
# IO

base_dir = '/raid/users/ebarnett/query/'
X     = np.load(os.path.join(base_dir, 'NWPU-RESISC45', 'fts/GeM_TVR50/X.npy'))
labs  = np.load(os.path.join(base_dir, 'NWPU-RESISC45', 'fts/GeM_TVR50/y.npy'))

p = np.random.permutation(X.shape[0])
X, labs = X[p], labs[p]

# --
# Postprocess

Xn = X.copy()
Xn = Xn / np.sqrt((Xn ** 2).sum(axis=-1, keepdims=True))
# !! Whitening breaks this badly -- why?

target_class   = labs[1]
y              = labs == target_class
act_prevalence = y.mean()

# --
# Run SimpleLAS

init_pt     = np.random.choice(np.where(y)[0], n_init, replace=False)
init_labels = {p:1 for p in init_pt}

las = SimpleLAS(Xn, init_labels=init_labels, pi=act_prevalence, eta=eta, alpha=alpha)

hits = [1]
idxs = list(init_labels.keys())

for i in trange(n_iter):
  idx = las.next_message
  las.set_label(idx, y[idx])
  
  idx = int(idx)
  
  hits.append(y[idx])
  idxs.append(idx)
  
  print(idx, y[idx])

avg = np.array(hits).cumsum()
_ = [plt.plot(np.cumsum(h), c='blue', alpha=0.2) for h in hist.values()]
_ = plt.plot(avg, c='red', alpha=0.5)
_ = plt.plot(np.arange(n_iter), c='grey', alpha=0.5)
_ = plt.plot(np.arange(n_iter) * act_prevalence, c='grey', alpha=0.5)
_ = plt.xlabel('# iterations')
_ = plt.ylabel('# hits')
_ = plt.title('mnist active search')
show_plot()

