#!/usr/bin/env python

"""
  simpleLAS.py
  
  Simpler version of activeSearchInterface.linearizedAS w/ reduced functionality
  
  !! Does not support sparse matrices yet
  !! Very little verbosity
  !! No `remove_self_degree`
  !! Dropped a bunch of functions from the API
  
  ```
    simp = SimpleLAS(X, init_labels=init_labels, pi=act_prev, eta=0.5, alpha=0.1)
    idx = simp.next_message
    simp.set_label(idx, Y[idx])
  ```
"""

from __future__ import print_function
import sys
import h5py
import numpy as np
from datetime import datetime

class SimpleLAS(object):
  
  def __init__(self, X, init_labels={}, pi=0.05, eta=0.5, w0=None, alpha=0, n=1, verbose=False):
    """
      Simple implementation of linearized active search
      
      X: data matrix (nobs rows, dim cols)
      init_labels: dictionary like {idx:lab}
      pi: ...
      eta: ...
      w0: ...
      alpha: ...
      n: number of candidate messages to return per iteration
        NB: Probably, any theoretical guarantees from the paper are degraded when n > 1,
        but in some applications it might be useful to get a set of proposals
    """
    
    # Init data
    
    
    X = X.T # !! For whatever reason, original implementation used 1 col per obs, which is nonstandard IMO
    
    self.X = X
    dim, n_obs = X.shape
    
    # Init params
    self.n              = n
    self.pi             = pi
    self.eta            = eta
    self.l              = (1 - self.eta) / self.eta
    self.alpha          = alpha
    self.w0             = w0 if w0 is not None else 1 / n_obs
    self.verbose        = verbose
    self.iter           = 0
    self.labels         = np.zeros(n_obs) - 1
    # self.hits           = [sum(init_labels.values())]
    self.labeled_idxs   = list(init_labels.keys())
    self.unlabeled_idxs = list(set(range(self.labels.shape[0])) - set(self.labeled_idxs))
    
    for idx in self.labeled_idxs:
      self.labels[idx] = init_labels[idx]
    
    self._init_matrices()
    
    if init_labels:
      self.next_message = self._nominate_next_messages()
    else:
      self.next_message = np.random.choice(range(n_obs), n, replace=False)

  def set_label(self, idx, lbl):
    assert((lbl == 0) or (lbl == 1))
    
    # Update labels
    self.iter += 1
    # self.hits.append(lbl)
    self.labels[idx] = lbl
    
    self.labeled_idxs.append(idx)
    self.unlabeled_idxs.remove(idx)
    
    # Update matrices
    self._update_matrices(idx, lbl)
    
    # Update nomination
    self.next_message = self._nominate_next_messages()
  
  def _nominate_next_messages(self):
    if self.n == 1:
      next_idx = (self.f + self.alpha * self.IM)[self.unlabeled_idxs].argmax()
      return np.array([self.unlabeled_idxs[next_idx]])
      
    else:
      tmp      = (self.f + self.alpha * self.IM)[self.unlabeled_idxs]
      next_idx = np.argpartition(tmp, -self.n)[-self.n:]
      next_idx = next_idx[np.argsort(tmp[next_idx])][::-1]
      
      if self.verbose:
        print(np.vstack([
          self.f[next_idx],
          self.alpha * self.IM[next_idx]
        ]).T)
      
      return np.array(self.unlabeled_idxs)[next_idx]
  
  def _init_matrices(self):
    X = self.X
    dim, n_obs = X.shape
    
    B          = np.where(self.labels == -1, 1 / (1 + self.w0), self.l / (1 + self.l))
    y_prime    = np.where(self.labels == -1, self.pi, self.labels)
    D          = (X.T.dot(X.dot(np.ones((n_obs,1))))).squeeze()
    self.Dinv  = 1. / D
    self.BDinv = (B * self.Dinv).squeeze()
    
    self.q     = (1 - B) * y_prime
    
    K         = np.eye(dim) - X.dot(self.BDinv[:,None] * X.T)
    self.Kinv = np.linalg.inv(K)
    
    self.f = self.q + self.BDinv * X.T.dot(self.Kinv.dot(X.dot(self.q)))
    
    if self.alpha > 0:
      self.dP   = (1. / self.l - self.w0) * D
      self.dPpi = (1. / self.l - self.pi * self.w0) * D
      self.z    = np.where(self.labels == -1, self.BDinv, 0)
      self.J    = np.squeeze(((self.Kinv.dot(self.X)) * self.X).sum(0))
      self.IM   = self._compute_IM()
    else:
      self.IM   = np.zeros(self.f.shape)
  
  def _update_matrices(self, idx, lbl):
    X = self.X
    
    self.BDinv[idx] = self.l / (1 + self.l) * self.Dinv[idx]
    
    self.q[idx] = lbl * 1 / (1 + self.l)
    
    gamma     = - (self.l / (1 + self.l) - 1 / (1 + self.w0)) * self.Dinv[idx]
    Xi        = X[:,idx]
    Kinv_Xi   = self.Kinv.dot(Xi)
    num       = Kinv_Xi.dot(Kinv_Xi.T)
    denom     = 1 + gamma * Xi.T.dot(Kinv_Xi)
    self.Kinv = self.Kinv - gamma * (num / denom)
    
    self.f = self.q + self.BDinv * X.T.dot(self.Kinv.dot(X.dot(self.q)))
    
    if self.alpha > 0:
      self.z[idx] = 0
      self.J      = self.J - (gamma / denom) * (np.squeeze(self.X.T.dot(Kinv_Xi)) ** 2)
      self.IM     = self._compute_IM()
  
  def _compute_IM(self):
    X = self.X
    
    Minv_u   = self.z + self.BDinv * X.T.dot(self.Kinv.dot(X.dot(self.z)))
    dpf      = self.dPpi - self.dP * self.f
    diagMi   = (1 + self.BDinv * self.J) * self.BDinv
    Df_tilde = dpf * diagMi / (1 + self.dP * diagMi)
    
    DF = (dpf - self.dP * Df_tilde) * Minv_u
    IM = self.f * (DF - Df_tilde)
    IM = IM * self.f.mean() / IM.mean()
    return IM.squeeze()
  