
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

class PyroCensoredNormal(ExponentialFamily, TorchDistributionMixin):
    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, censoring, validate_args=None):
        
        self.loc, self.scale, self.censoring = broadcast_all(loc, scale, censoring)
        
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(censoring, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(PyroCensoredNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroCensoredNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.censoring = self.censoring
        super(PyroCensoredNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        self.threshold = self.censoring==1
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        log_prob = -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

        log_prob[self.threshold] = math.log(1 - self.cdf(value)[self.threshold] + 0.01) if isinstance(1 - self.cdf(value)[self.threshold] + 1e-6,
                                                                                                                       Number) else (1 - self.cdf(value)[self.threshold] + 1e-6).log()
        return log_prob

    def cdf(self, value):
        
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

class CensoredHomoscedGaussian(Likelihood):
    
    def __init__(self, variance=None, censoring=None):
        super(CensoredHomoscedGaussian, self).__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)
        self.censoring = censoring

    def forward(self, f_loc, y=None):
       # y_var = f_var + self.variance
        y_var = self.variance
        y_dist = PyroCensoredNormal(loc=f_loc, scale=y_var.sqrt(), censoring=self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
    
class HomoscedGaussian(Likelihood):
    
    def __init__(self, variance=None):
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, f_loc, y=None):
        
        y_var = self.variance

        y_dist = dist.Normal(f_loc, y_var.sqrt())
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)

class CensoredHeteroGaussian(Likelihood):
    
    def __init__(self, variance=None, censoring=None):
        super(CensoredHeteroGaussian, self).__init__()
        self.censoring = censoring

    def forward(self, f, g, y=None):
        y_dist = PyroCensoredNormal(f, torch.exp(g),self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
    
class PyroCensoredPois(ExponentialFamily, TorchDistributionMixin):
    
    arg_constraints = {'rate': constraints.positive}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def __init__(self, rate, censoring, validate_args=None):
        self.rate, self.censoring = broadcast_all(rate, censoring)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super(PyroCensoredPois, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroCensoredPois, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        new.censoring = self.censoring
        super(PyroCensoredPois, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))


    def log_prob(self, value):
        self.threshold = self.censoring==1
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        log_prob = (rate.log() * value) - rate - (value + 1).lgamma()
        log_prob[self.threshold] = math.log(1 - self.cdf(value)[self.threshold] + 0.01) if isinstance(1 - self.cdf(value)[self.threshold] + 1e-6, Number) else (1 - self.cdf(value)[self.threshold] + 1e-6).log()
        return log_prob

    def cdf(self, value):
        cdf_out = []
        if self._validate_args:
            self._validate_sample(value)
            
        if value.shape == torch.Size([]):
            summer = 0
            
            for i in range(int(value)+1):
                summer += (self.rate**i)/(torch.tensor(i+1,dtype=torch.float).lgamma().exp())
            return (-self.rate).exp()*summer
        else:
            for j,val in enumerate(value):
                summer = 0
                for i in range(int(val)+1):
                    summer += (self.rate[j]**i)/(torch.tensor(i+1,dtype=torch.float).lgamma().exp())
                cdf_out.append(summer)
            cdf_out = torch.tensor(cdf_out)   
            return (-self.rate).exp()*cdf_out
       
    @property
    def _natural_params(self):
        return (torch.log(self.rate), )

    def _log_normalizer(self, x):
        return torch.exp(x)

class Censored_Poisson(Likelihood):
    
    def __init__(self, variance=None, censoring=None, response_function=None):
        super().__init__()
        
        self.response_function = torch.exp if response_function is None else response_function
        self.censoring = censoring
        
    def forward(self, f_loc, y=None):
        
        f_res = self.response_function(f_loc)

        y_dist = PyroCensoredPois(rate=f_res, censoring=self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
    
class Poisson(Likelihood):
    
    def __init__(self, variance=None, response_function=None):
        super().__init__()
        
        self.response_function = torch.exp if response_function is None else response_function
        
    def forward(self, f_loc, y=None):
        
        f_res = self.response_function(f_loc)

        y_dist = dist.Poisson(f_res)
        
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
