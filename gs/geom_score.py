from __future__ import absolute_import
from __future__ import print_function
from .utils import relative
from .utils import witness
import numpy as np
from . import barycenter

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import torch 

def rlt(X, L_0=64, gamma=None, i_max=100):
    """
      This function implements Algorithm 1 for one sample of landmarks.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing RLT(i, 1, X, L)
      for randomly sampled landmarks.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError('X should be a numpy array')
    if len(X.shape) != 2:
        raise ValueError('X should be 2d array, got shape {}'.format(X.shape))
    N = X.shape[0]
    if gamma is None:
        gamma = 1.0 / 128 * N / 5000
    I_1, alpha_max = witness(X, L_0=L_0, gamma=gamma)
    res = relative(I_1, alpha_max, i_max=i_max)
    return res


def rlts(X, L_0=64, gamma=None, i_max=100, n=1000):
    """
      This function implements Algorithm 1.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.
      n: int, number of samples
    Returns
      An array of size (n, i_max) containing RLT(i, 1, X, L)
      for n collections of randomly sampled landmarks.
    """
    rlts = np.zeros((n, i_max))
    for i in range(n):
        rlts[i, :] = rlt(X, L_0, gamma, i_max)
        if i % 10 == 0:
            print('Done {}/{}'.format(i, n))
    return rlts


def dist(n):
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    res = cdist(x, x, metric='sqeuclidean')
    return res

def barymean(rlts):
    D = dist(len(rlts))
    device = 'cuda'
    D /= D.max()
    P = torch.from_numpy(rlts.T).double().to(device)
    C = torch.from_numpy(D).double().to(device)
    mrlt, _ = barycenter.compute(P=P, C=C)
    return mrlt.cpu().numpy()

def geom_score(rlts1, rlts2, dist_type='wasserstein'):
    """
      This function implements Algorithm 2.
    Args:
       rlts1 and rlts2: arrays as returned by the function "rlts".
    Returns
       Float, a number representing topological similarity of two datasets.
    """
    # mrlt1 = barymean(rlts1)
    # mrlt2 = barymean(rlts2)
    mrlt1 = rlts1
    mrlt2 = rlts2
    if dist_type == 'wasserstein':
      return wasserstein_distance(mrlt1, mrlt2)
    elif dist_type == 'euclidean':
      return np.sum((mrlt1 - mrlt2) ** 2)
