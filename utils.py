
# coding: utf-8

# In[1]:


import numpy as np
import torch
import copy
from sklearn.preprocessing import StandardScaler

def log_error_f(f):
    return torch.log(1+torch.exp(f))

def create_1d_data(N):
    
    #np.random.seed(10)
    #torch.manual_seed(49)
    X1 = np.linspace(0, 10, N)
    y = 0.5*np.sin(2*X1) + 2 + X1/10

    # Generate noisy observations
    noise_scale = torch.linspace(0,0.3,N)*1
    epsilon = torch.distributions.Normal(loc=0, scale=noise_scale).sample()
    Y1 = y + epsilon.numpy()

    return X1, Y1, y,noise_scale

def censor_1d_data(X1_dat,Y1_dat,y_dat,int_low):
    #np.random.seed(10)
    #torch.manual_seed(49)
    int_high = int_low + 0.1
    Y1_cens = copy.deepcopy(Y1_dat)

    # Select random points as censored and apply censoring
    censoring = np.int32(0.5*np.sin(2*X1_dat) + 2 >= 2) 
    p_c = np.random.uniform(low=int_low, high=int_high, size=np.sum(censoring==1))
    Y1_cens[censoring == 1] = Y1_dat[censoring == 1]*(1-p_c)

    X1_dat = (X1_dat-X1_dat.mean())/(X1_dat.std())

    # Standardize
    scaler = StandardScaler()
    Y1_cens_sc = scaler.fit_transform(Y1_cens.reshape(-1,1))
    y_sc = scaler.transform(y_dat.reshape(-1,1))
    
    X1_dat = torch.as_tensor(X1_dat).reshape(-1)
    Y1_dat = torch.as_tensor(Y1_dat).reshape(-1)
    y_sc = torch.as_tensor(y_sc).reshape(-1)
    Y1_cens_sc = torch.as_tensor(Y1_cens_sc).reshape(-1)
    
    return X1_dat,Y1_dat,y_sc,Y1_cens_sc,censoring

def create_2nd_data(N):
    #np.random.seed(10)
    #torch.manual_seed(49)
    X2 = torch.linspace(0,10,N)
    y2 = torch.sin(2*X2)
    y2 = (y2-y2.mean())/y2.std()

    noise_scale_y2 = torch.tensor(np.flip(np.linspace(0,0.4,N))*1)
    epsilon_y2 = torch.distributions.Normal(loc=0, scale=noise_scale_y2).sample()
    
    X2 = (X2-X2.mean())/(X2.std())
    
    Y2 = y2 + epsilon_y2
    Y2_mean = Y2.mean()
    Y2_std = Y2.std()
    Y2 = (Y2-Y2_mean)/Y2_std
    
    return X2,Y2,y2,noise_scale_y2

