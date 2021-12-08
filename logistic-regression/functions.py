import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.special import expit
from copy import deepcopy
from scipy.stats import bernoulli

def randk(x, args):
    k = args[1]
    n = len(x)
    indices = np.random.choice(np.arange(n), replace=False, size=(n - k))
    x[indices] = 0
    return x, 32 * k

def identical(x, args):
    n = len(x)
    return x, 32*n

def topk(x, args):
    k = args[1]
    k_order_statistics = np.sort(np.abs(x))[-k]
    return np.where(np.abs(x) >= k_order_statistics, x, 0), 32*k
        
    
def threshold(x, args):
    thr = args[1]
    sparse_vector = np.where(np.abs(x) >= thr, x, 0)
    return sparse_vector, 32*np.sum(sparse_vector!=0)

def logreg_loss(x, args):
    A = args[0]
    y = args[1]
    l2 = args[2]
    sparse = args[3]
    assert l2 >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    degree1 = np.zeros(A.shape[0])
    if sparse == True:
        degree2 = -(A * x) * y
        l = np.logaddexp(degree1, degree2)
    else:
        degree2 = -A.dot(x) * y
        l = np.logaddexp(degree1, degree2)
    m = y.shape[0]
    return np.sum(l) / m + l2/2 * norm(x) ** 2

def logreg_grad(x, args):
    A = args[0]
    y = args[1]
    mu = args[2]
    sparse = args[3]
    assert mu >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    if sparse == True:
        degree = -y * (A * x)
        sigmas = expit(degree)
        loss_grad = -A.transpose() * (y * sigmas) / A.shape[0]
    else:
        degree = -y * (A.dot(x))
        sigmas = expit(degree)
        loss_grad = -A.T.dot(y * sigmas) / A.shape[0]
    assert len(loss_grad) == len(x)
    return loss_grad + mu * x

def logreg_grad_plus_lasso(x, args):
    return logreg_grad(x, args) + args[4] * np.sign(x)

def r(x, l1):
    assert (l1 >= 0)
    return l1 * norm(x, ord = 1)

def F(x, args):
    return logreg_loss(x, args) + r(x, args[4])
    
def prox_R(x, l1):
    res = np.abs(x) - l1 * np.ones(len(x))
    return np.where(res > 0, res, 0) * np.sign(x)