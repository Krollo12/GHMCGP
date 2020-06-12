
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

from GP_models import GPModel, VariationalGP, VariationalMGP, HGPModel, VariationalMHGP
from GP_likelihoods import PyroCensoredNormal, CensoredHomoscedGaussian, HomoscedGaussian, CensoredHeteroGaussian

def _zero_mean_function(x):
    return 0

def standard_gp_train(X1,Y1,Y1_cens_sc,y,y_sc,censoring,int_low,noise_scale,file):
    
    pyro.clear_param_store()
    kern = gp.kernels.RBF(input_dim=1,active_dims=[0],lengthscale=torch.tensor(1.),variance=torch.tensor(1.))
    like = HomoscedGaussian()
    sgphomo = VariationalGP(X=X1, y=Y1_cens_sc, kernel=kern, likelihood=like, mean_function=None,latent_shape=None, whiten=False,jitter=0.005)
    guide = AutoMultivariateNormal(sgphomo.model)
    
    optimizer = pyro.optim.ClippedAdam({"lr": 0.003,"lrd": 0.99969})
    svi = SVI(sgphomo.model, guide, optimizer, Trace_ELBO(num_particles=40))
    
    num_epochs = 4000
    losses = []
    pyro.clear_param_store()
    for epoch in range(num_epochs):
        loss = svi.step(X1, Y1_cens_sc)
        losses.append(loss)
        if epoch==num_epochs-1:
            
            with torch.no_grad():
                predictive = Predictive(sgphomo.model, guide=guide, num_samples=1000,
                return_sites=("f", "g", "_RETURN"))
                samples = predictive(X1)
                f_samples = samples["f"]
                f_mean = f_samples.mean(dim=0)
                f_std = sgphomo.likelihood.variance.sqrt().item()
                f_025 = np.quantile(a=f_samples.detach().numpy(), q=0.025, axis=0)
                f_975 = np.quantile(a=f_samples.detach().numpy(), q=0.975, axis=0)
                fig = plt.figure(figsize=(20,6))
                fig.add_subplot(121)
                plt.plot(X1.numpy(), y_sc.numpy(), linestyle="--", color="black")
                plt.plot(X1.detach().numpy(), f_mean.detach().numpy(), color="black")
                plt.fill_between(X1.detach().numpy(), f_mean.detach().numpy() - 1.96*f_std, f_mean.detach().numpy() + 1.96*f_std, alpha=0.3)
                plt.fill_between(X1.reshape(-1).detach().numpy(), f_025, f_975, alpha=0.3,label='Mean uncertainty')
                plt.scatter(X1.numpy()[censoring==1].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==1].reshape(-1,1), marker="x", label="Censored Observations", color='#348ABD')
                plt.scatter(X1.numpy()[censoring==0].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==0].reshape(-1,1), marker="o", label="Non-Censored Observations", color='#348ABD')
                plt.legend(prop={'size': 12})

                fig.add_subplot(122)
                plt.plot(np.arange(len(noise_scale)), np.ones(len(noise_scale))*sgphomo.likelihood.variance.sqrt().item(),label='Estimated noise')
                plt.plot(np.arange(len(noise_scale)), noise_scale,label='True noise scale')
                plt.legend(prop={'size': 12})
                plt.savefig('Experiments/Synthetic/SGP/SGP_Synthetic_{}.png'.format(int_low))

    fig1 = plt.figure(figsize=(8,6))
    plt.plot(losses,label='Loss')
    plt.legend(prop={'size': 12})
    plt.savefig('Experiments/Synthetic/SGP/SGP_Synthetic_Loss_{}.png'.format(int_low))
    
    
    RMSE = sqrt(mean_squared_error(y_sc, f_mean.detach().numpy()))
    NLPD = -(1/len(Y1)*sgphomo.likelihood.y_dist.log_prob(y_sc).sum().item())
    
    file.write('\n Intensity :' + str(int_low) + ' ')  
    file.write('RMSE: ' + str(RMSE)+ ' ') 
    file.write('NLPD: ' + str(NLPD))
    
def censored_gp_train(X1,Y1,Y1_cens_sc,y,y_sc,censoring,int_low,noise_scale,file):
    
    pyro.clear_param_store()
    kern = gp.kernels.RBF(input_dim=1,active_dims=[0],lengthscale=torch.tensor(1.),variance=torch.tensor(1.))
    like_cens = CensoredHomoscedGaussian(censoring=torch.tensor(censoring))
    sgphomo_cens = VariationalGP(X=X1, y=Y1_cens_sc, kernel=kern, likelihood=like_cens, mean_function=None,
                     latent_shape=None, whiten=False,jitter=0.005)
    guide = AutoMultivariateNormal(sgphomo_cens.model)
    
    optimizer = pyro.optim.ClippedAdam({"lr": 0.003,"lrd": 0.99969})
    svi = SVI(sgphomo_cens.model, guide, optimizer, Trace_ELBO(num_particles=40))
    
    num_epochs = 4000
    losses = []
    pyro.clear_param_store()
    for epoch in range(num_epochs):
        loss = svi.step(X1, Y1_cens_sc)
        losses.append(loss)
        if epoch==num_epochs-1:
            
            with torch.no_grad():
                predictive = Predictive(sgphomo_cens.model, guide=guide, num_samples=1000,
                return_sites=("f", "g", "_RETURN"))
                samples = predictive(X1)
                f_samples = samples["f"]
                f_mean = f_samples.mean(dim=0)
                f_std = sgphomo_cens.likelihood.variance.sqrt().item()
                f_025 = np.quantile(a=f_samples.detach().numpy(), q=0.025, axis=0)
                f_975 = np.quantile(a=f_samples.detach().numpy(), q=0.975, axis=0)
                fig = plt.figure(figsize=(20,6))
                fig.add_subplot(121)
                plt.plot(X1.numpy(), y_sc.numpy(), linestyle="--", color="black")
                plt.plot(X1.detach().numpy(), f_mean.detach().numpy(), color="black")
                plt.fill_between(X1.detach().numpy(), f_mean.detach().numpy() - 1.96*f_std, f_mean.detach().numpy() + 1.96*f_std, alpha=0.3)
                plt.fill_between(X1.reshape(-1).detach().numpy(), f_025, f_975, alpha=0.3,label='Mean uncertainty')
                plt.scatter(X1.numpy()[censoring==1].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==1].reshape(-1,1), marker="x", label="Censored Observations", color='#348ABD')
                plt.scatter(X1.numpy()[censoring==0].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==0].reshape(-1,1), marker="o", label="Non-Censored Observations", color='#348ABD')
                plt.legend(prop={'size': 12})

                fig.add_subplot(122)
                plt.plot(np.arange(len(noise_scale)), np.ones(len(noise_scale))*sgphomo_cens.likelihood.variance.sqrt().item(),label='Estimated noise')
                plt.plot(np.arange(len(noise_scale)), noise_scale,label='True noise scale')
                plt.legend(prop={'size': 12})
                plt.savefig('Experiments/Synthetic/CGP/CGP_Synthetic_{}.png'.format(int_low))

    fig1 = plt.figure(figsize=(8,6))
    plt.plot(losses,label='Loss')
    plt.legend(prop={'size': 12})
    plt.savefig('Experiments/Synthetic/CGP/CGP_Synthetic_Loss_{}.png'.format(int_low))
    
    
    RMSE = sqrt(mean_squared_error(y_sc, f_mean.detach().numpy()))
    NLPD = -(1/len(Y1)*sgphomo_cens.likelihood.y_dist.log_prob(y_sc).sum().item())
    
    file.write('\n Intensity :' + str(int_low) + ' ')  
    file.write('RMSE: ' + str(RMSE)+ ' ') 
    file.write('NLPD: ' + str(NLPD))
    
def multi_gp_train(X_augmented,Y_augmented,X1,Y1,X2,Y2,Y1_cens_sc,y,y2,y_sc,censoring,censoring_mul,int_low,noise_scale,file):
    N1 = len(y)
    pyro.clear_param_store()
    k1 = gp.kernels.RBF(input_dim=1,active_dims=[0],lengthscale=torch.tensor(1.),variance=torch.tensor(1.))
    coreg = gp.kernels.Coregionalize(input_dim=X_augmented.shape[1], rank=1)
    f_rbf = gp.kernels.Product(k1, coreg)

    like = CensoredHomoscedGaussian(censoring=censoring_mul)
    multi_hom_gp = VariationalMGP(X=X_augmented, y=Y_augmented, kernel=f_rbf, likelihood=like, jitter=0.005)
    guide = AutoMultivariateNormal(multi_hom_gp.model)
    
    optimizer = pyro.optim.ClippedAdam({"lr": 0.003,"lrd": 0.99969})
    svi = SVI(multi_hom_gp.model, guide, optimizer, Trace_ELBO(num_particles=60))
    
    num_epochs = 12000
    losses = []
    pyro.clear_param_store()
    for epoch in range(num_epochs):
        loss = svi.step(X_augmented, Y_augmented)
        losses.append(loss)
        if epoch==num_epochs-1:
            
            with torch.no_grad():
                predictive = Predictive(multi_hom_gp.model, guide=guide, num_samples=1000,
                return_sites=("f", "g", "_RETURN"))
                samples = predictive(X_augmented)
                f_samples = samples["f"]
                f_mean = f_samples.mean(dim=0)
                f_std = multi_hom_gp.likelihood.variance.sqrt().item()
                f_025 = np.quantile(a=f_samples.detach().numpy(), q=0.025, axis=0)
                f_975 = np.quantile(a=f_samples.detach().numpy(), q=0.975, axis=0)
                fig = plt.figure(figsize=(20,12))
                fig.add_subplot(221)
                plt.plot(X1.numpy(), y_sc.numpy(), linestyle="--", color="black")
                plt.plot(X1.reshape(-1).detach().numpy(), f_mean.detach().numpy()[0:(N1)], color="black")
                plt.fill_between(X1.reshape(-1).detach().numpy(), f_mean.detach().numpy()[0:(N1)] - 1.96*f_std, f_mean.detach().numpy()[0:(N1)] + 1.96*f_std, alpha=0.3)
                plt.fill_between(X1.reshape(-1).detach().numpy(), f_025[0:N1], f_975[0:N1], alpha=0.3,label='Y1 Mean uncertainty')
                plt.scatter(X1.numpy()[censoring==1].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==1].reshape(-1,1), marker="x", label="Censored Observations", color='#348ABD')
                plt.scatter(X1.numpy()[censoring==0].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==0].reshape(-1,1), marker="o", label="Non-Censored Observations", color='#348ABD')
                plt.legend(prop={'size': 12})
                
                
                
                fig.add_subplot(222)
                plt.plot(X2.numpy(), y2.numpy(), linestyle="--", color="gray")
                plt.plot(X2.reshape(-1).detach().numpy(), f_mean.detach().numpy()[(N1):], color="black")
                plt.fill_between(X2.reshape(-1).detach().numpy(), f_mean.detach().numpy()[(N1):] - 1.96*f_std, f_mean.detach().numpy()[(N1):] + 1.96*f_std, alpha=0.3)
                plt.fill_between(X2.reshape(-1).detach().numpy(), f_025[N1:], f_975[N1:], alpha=0.3,label='Y2 mean uncertainty')
                plt.scatter(X2.numpy(), Y2.numpy(),label='Y2 Observed values')
                plt.legend(prop={'size': 12})
                
                
                fig.add_subplot(223)
                plt.plot(np.arange(len(noise_scale)), np.ones(len(noise_scale))*f_std,'b--',label='Y1 inference noise')
                plt.plot(np.arange(len(noise_scale)), noise_scale,'b-',label='Y1 true noise')
                plt.legend(prop={'size': 12})
                
                fig.add_subplot(224)
                plt.plot(np.arange(len(noise_scale_y2)), np.ones(len(noise_scale))*f_std,'r--',label='Y2 inference noise')
                plt.plot(np.arange(len(noise_scale_y2)), noise_scale_y2,'r-',label='Y2 true noise')
                plt.legend(prop={'size': 12})
                plt.savefig('Experiments/Synthetic/MGP/MGP_Synthetic_{}.png'.format(int_low))

    fig1 = plt.figure(figsize=(8,6))
    plt.plot(losses,label='Loss')
    plt.legend(prop={'size': 12})
    plt.savefig('Experiments/Synthetic/MGP/MGP_Synthetic_Loss_{}.png'.format(int_low))
    
    
    RMSE = sqrt(mean_squared_error(y_sc, f_mean.detach().numpy()[:(N1)]))
    NLPD = -(1/len(y)*multi_hom_gp.likelihood.y_dist.log_prob(torch.cat((y_sc,y2.type(torch.float64))))[:(len(y_sc))].sum().item())
    
    file.write('\n Intensity :' + str(int_low) + ' ')  
    file.write('RMSE: ' + str(RMSE)+ ' ') 
    file.write('NLPD: ' + str(NLPD))
    
def hetero_multi_gp_train(X_augmented,Y_augmented,X1,Y1,X2,Y2,Y1_cens_sc,y,y2,y_sc,censoring,censoring_mul,int_low,noise_scale,file):
    N1 = len(y)
    pyro.clear_param_store()
    k1 = gp.kernels.RBF(input_dim=1,active_dims=[0],lengthscale=torch.tensor(1.),variance=torch.tensor(1.))
    coreg = gp.kernels.Coregionalize(input_dim=X_augmented.shape[1], rank=1)
    f_rbf = gp.kernels.Product(k1, coreg)

    g0_rbf = gp.kernels.RBF(input_dim=1, lengthscale=torch.tensor(1.), variance=torch.tensor(1.))
    g1_rbf = gp.kernels.RBF(input_dim=1, lengthscale=torch.tensor(1.), variance=torch.tensor(1.))

    like = CensoredHeteroGaussian(censoring=censoring_mul)
    multi_het_gp = VariationalMHGP(X=X_augmented, y=Y_augmented, f_kernel=f_rbf, g0_kernel=g0_rbf, g1_kernel=g1_rbf, likelihood=like, jitter=0.005)

    guide = AutoMultivariateNormal(multi_het_gp.model)
    
    optimizer = pyro.optim.ClippedAdam({"lr": 0.003,"lrd": 0.99969})
    svi = SVI(multi_het_gp.model, guide, optimizer, Trace_ELBO(num_particles=60))
    
    num_epochs = 12000
    losses = []
    pyro.clear_param_store()
    for epoch in range(num_epochs):
        loss = svi.step(X_augmented, Y_augmented)
        losses.append(loss)
        if epoch==num_epochs-1:
            
            with torch.no_grad():
                predictive = Predictive(multi_het_gp.model, guide=guide, num_samples=1000, return_sites=("f","g0","g1", "_RETURN"))
                samples = predictive(X_augmented)
                f_samples = samples["f"]
                f_mean = f_samples.mean(dim=0)
                g0_samples = samples["g0"]
                g1_samples = samples["g1"]
                g0_mean = g0_samples.mean(dim=0).detach().numpy()
                g1_mean = g1_samples.mean(dim=0).detach().numpy()
                f_025 = np.quantile(a=f_samples.detach().numpy(), q=0.025, axis=0)
                f_975 = np.quantile(a=f_samples.detach().numpy(), q=0.975, axis=0)
                
                fig = plt.figure(figsize=(20,12))
                fig.add_subplot(221)
                plt.plot(X1.numpy(), y_sc.numpy(), linestyle="--", color="black")
                plt.plot(X1.reshape(-1).detach().numpy(), f_mean.detach().numpy()[0:(N1)], color="black")
                plt.fill_between(X1.reshape(-1).detach().numpy(), f_mean.detach().numpy()[0:(N1)] - 1.96*np.exp(g0_mean), f_mean.detach().numpy()[0:(N1)] + 1.96*np.exp(g0_mean), alpha=0.3)
                plt.fill_between(X1.reshape(-1).detach().numpy(), f_025[0:N1], f_975[0:N1], alpha=0.3,label='Y1 Mean uncertainty')
                plt.scatter(X1.numpy()[censoring==1].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==1].reshape(-1,1), marker="x", label="Censored Observations", color='#348ABD')
                plt.scatter(X1.numpy()[censoring==0].reshape(-1,1), y=Y1_cens_sc.numpy()[censoring==0].reshape(-1,1), marker="o", label="Non-Censored Observations", color='#348ABD')
                plt.legend(prop={'size': 12})
                
                
                
                fig.add_subplot(222)
                plt.plot(X2.numpy(), y2.numpy(), linestyle="--", color="gray")
                plt.plot(X2.reshape(-1).detach().numpy(), f_mean.detach().numpy()[(N1):], color="black")
                plt.fill_between(X2.reshape(-1).detach().numpy(), f_mean.detach().numpy()[(N1):] - 1.96*np.exp(g1_mean), f_mean.detach().numpy()[(N1):] + 1.96*np.exp(g1_mean), alpha=0.3)
                plt.fill_between(X2.reshape(-1).detach().numpy(), f_025[N1:], f_975[N1:], alpha=0.3,label='Y2 mean uncertainty')
                plt.scatter(X2.numpy(), Y2.numpy(),label='Y2 Observed values')
                plt.legend(prop={'size': 12})
                
                
                fig.add_subplot(223)
                plt.plot(np.arange(len(g0_mean)), np.exp(g0_mean),'b--',label='Y1 inference noise')
                plt.plot(np.arange(len(noise_scale)), noise_scale,'b-',label='Y1 true noise')
                plt.legend(prop={'size': 12})
                
                fig.add_subplot(224)
                plt.plot(np.arange(len(g1_mean)), np.exp(g1_mean),'r--',label='Y2 inference noise')
                plt.plot(np.arange(len(noise_scale_y2)), noise_scale_y2,'r-',label='Y2 true noise')
                plt.legend(prop={'size': 12})
                plt.savefig('Experiments/Synthetic/HMGP/HMGP_Synthetic_{}.png'.format(int_low))

    fig1 = plt.figure(figsize=(8,6))
    plt.plot(losses,label='Loss')
    plt.legend(prop={'size': 12})
    plt.savefig('Experiments/Synthetic/HMGP/HMGP_Synthetic_Loss_{}.png'.format(int_low))
    
    
    RMSE = sqrt(mean_squared_error(y_sc, f_mean.detach().numpy()[:(N1)]))
    NLPD = -(1/len(y_sc)*multi_het_gp.likelihood.y_dist.log_prob(torch.cat((y_sc,y2.type(torch.float64))))[:(len(y_sc))].sum().item())
    
    file.write('\n Intensity :' + str(int_low) + ' ')  
    file.write('RMSE: ' + str(RMSE)+ ' ') 
    file.write('NLPD: ' + str(NLPD))
    

