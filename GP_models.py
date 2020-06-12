
# coding: utf-8

# In[ ]:


import os
import numpy as np
import scipy.io
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
import math
from sklearn.metrics import mean_squared_error
import copy
from sklearn.preprocessing import StandardScaler
from math import sqrt
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.nn.module import PyroParam
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions.exp_family import ExponentialFamily
from pyro.contrib.gp.parameterized import Parameterized
from numbers import Number
from torch.distributions import constraints
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal,AutoLowRankMultivariateNormal
from pyro.infer import SVI, EmpiricalMarginal, TracePosterior, Trace_ELBO, Predictive,TraceMeanField_ELBO,TraceGraph_ELBO,TracePredictive

def _zero_mean_function(x):
    return 0

class GPModel(Parameterized):
    def __init__(self, X, y, kernel, mean_function=None, jitter=1e-6):
        super().__init__()
        self.set_data(X, y)
        self.kernel = kernel
        #self.censoring = censoring
        self.mean_function = (mean_function if mean_function is not None else
                              _zero_mean_function)
        self.jitter = jitter

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def forward(self, Xnew, full_cov=False):
        raise NotImplementedError

    def set_data(self, X, y=None):
        
        if y is not None and X.size(0) != y.size(-1):
            raise ValueError("Expected the number of input data points equal to the "
                             "number of output data points, but got {} and {}."
                             .format(X.size(0), y.size(-1)))
        self.X = X
        self.y = y

    def _check_Xnew_shape(self, Xnew):
        
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.X.dim(), Xnew.dim()))
        if self.X.shape[1:] != Xnew.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.X.shape[1:], Xnew.shape[1:]))
            
class VariationalGP(GPModel):
    
    def __init__(self, X, y, kernel, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood
       
        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        self.whiten = whiten
        self._sample_latent = True

    def model(self, X, y=None):
        self.set_mode("model")

        N = X.size(0)
        Kff = self.kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky()
        
        zero_loc = torch.zeros_like(X)
        f = pyro.sample("f", dist.MultivariateNormal(zero_loc, scale_tril=Lff).to_event(zero_loc.dim() - 1))
        if y is None:
            return f
        else:
            return self.likelihood(f, y)
        
class VariationalMGP(GPModel):
    
    def __init__(self, X, y, kernel, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6):
        super(VariationalMGP, self).__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        self.whiten = whiten
        self._sample_latent = True

    def model(self, X, y=None):
        self.set_mode("model")

        N = X.size(0)
        
        Kff = self.kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky()
        
        if len(X.shape)>1:
            zero_loc_f = torch.zeros_like(X[:,0])
        else:
            zero_loc = torch.zeros_like(X)
        
        f = pyro.sample("f", dist.MultivariateNormal(zero_loc_f, scale_tril=Lff))
        
        if y is None:
            return f
        else:
            return self.likelihood(f, y)
        
        
class HGPModel(Parameterized):

    def __init__(self, X, y, f_kernel, g0_kernel, g1_kernel, mean_function=None, jitter=1e-6):
        super(HGPModel, self).__init__()
        self.set_data(X, y)
        self.f_kernel = f_kernel
        self.g0_kernel = g0_kernel
        self.g1_kernel = g1_kernel
        self.mean_function = (mean_function if mean_function is not None else
                              _zero_mean_function)
        self.jitter = jitter

    def model(self):
        
        raise NotImplementedError

    def guide(self):
        
        raise NotImplementedError

    def forward(self, Xnew, full_cov=False):
        
        raise NotImplementedError

    def set_data(self, X, y=None):
        
        if y is not None and X.size(0) != y.size(-1):
            raise ValueError("Expected the number of input data points equal to the "
                             "number of output data points, but got {} and {}."
                             .format(X.size(0), y.size(-1)))
        self.X = X
        self.y = y

    def _check_Xnew_shape(self, Xnew):
       
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.X.dim(), Xnew.dim()))
        if self.X.shape[1:] != Xnew.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.X.shape[1:], Xnew.shape[1:]))
            
class VariationalMHGP(HGPModel):
    
    def __init__(self, X, y, f_kernel, g0_kernel, g1_kernel , likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6):
        super(VariationalMHGP, self).__init__(X, y, f_kernel, g0_kernel,g1_kernel, mean_function, jitter)

        self.likelihood = likelihood

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        self.whiten = whiten
        self._sample_latent = True

    def model(self, X, y=None):
        self.set_mode("model")

        N = X.size(0)
        N1 = sum(X[:,1]==1)
        N2 = sum(X[:,2]==1)
        
        Kff = self.f_kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky()
        
        Kgg0 = self.g0_kernel(X[:(N1),:]).contiguous()
        Kgg0.view(-1)[::(N1) + 1] += self.jitter  # add jitter to the diagonal
        Lgg0 = Kgg0.cholesky()
        
        Kgg1 = self.g1_kernel(X[(N1):,:]).contiguous()
        Kgg1.view(-1)[::(N2) + 1] += self.jitter  # add jitter to the diagonal
        Lgg1 = Kgg1.cholesky()
        
        if len(X.shape)>1:
            zero_loc_f = torch.zeros_like(X[:,0])
            zero_loc_g0 = torch.zeros_like(X[:N1,0])
            zero_loc_g1 = torch.zeros_like(X[:N2,0])
        else:
            zero_loc = torch.zeros_like(X)
        
        f = pyro.sample("f", dist.MultivariateNormal(zero_loc_f, scale_tril=Lff))
        g0 = pyro.sample("g0", dist.MultivariateNormal(zero_loc_g0, scale_tril=Lgg0))
        g1 = pyro.sample("g1", dist.MultivariateNormal(zero_loc_g1, scale_tril=Lgg1))
        g = torch.cat((g0, g1), 0)
        
        if y is None:
            return f
        else:
            return self.likelihood(f, g, y)
        
class VariationalPoisGP(GPModel):
    
    def __init__(self, X, y, kernel, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood
        
        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        self.whiten = whiten
        self._sample_latent = True

    def model(self, X, y=None):
        self.set_mode("model")

        N = X.size(0)
        Kff = self.kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky()
        zero_loc = torch.zeros_like(X)
        f = pyro.sample("f", dist.MultivariateNormal(zero_loc, scale_tril=Lff).to_event(zero_loc.dim() - 1))
        if y is None:
            return f
        else:
            return self.likelihood(f, y)

