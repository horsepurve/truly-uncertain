import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import time
import sys
import pickle
import platform
import copy

#%%
def w_l(values, v_mean):
    '''
    '''
    dis = np.sum((values - v_mean) ** 2, axis=-1)
    S   = np.sum(dis)
    w   = np.log(S / dis)
    return w / np.sum(w)
#%%
def see_v_s(v_s):
    '''
    Parameters
    ----------
    v_s : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    L = len(v_s)
    diffs = []        
    for i in range(L-1):
        diff = np.sum((v_s[i+1] - v_s[i]) ** 2, axis=-1)
        diffs.append(diff)
    return diffs
#%%
def see_L(values, v_, w_l):
    dis = np.sum((values - v_mean) ** 2, axis=-1)
    return np.sum(dis*w_l)
#%%
def v_project1(v_, ind_mean):
    iis  = []
    dist = []
    
    for ii in range(v_.shape[0]):
        if ind_mean == ii:
            continue
        iis.append(ii)
        dist.append(np.abs(v_[ind_mean] - v_[ii])/1.41421)
    
    iis          = np.array(iis)
    dist         = np.array(dist)
    
    target_plane = iis[np.argmin(dist)]
    
    update       = 0.5*(v_[ind_mean] + v_[target_plane])
    v_[ind_mean]     = update + 1e-6
    v_[target_plane] = update - 1e-6
    
    return v_
#%%
def v_project(v_, ind_mean, C=100):
    v__   = copy.deepcopy(v_)
    v_out = copy.deepcopy(v_)
    
    for ii in range(v_.shape[0]):
        if ind_mean == ii:
            continue
        update        = 0.5*(v_[ind_mean] + v_[ii])
        v__[ind_mean] = update
        v__[ii]       = update
        
        if sum(update >= v__) == C:
            # print('here:', ii, ind_mean, np.argmax(v_))
            v_out[ind_mean] = update + 1e-4
            v_out[ii]       = update - 1e-4
    
    return v_out
#%%
def v_project2(v_, ind_mean, C=100):
    ind_list = np.argsort(v_)[::-1]
    
    for iteration in range(C):
        '''
        ind_mean is our target
        '''
        sub_ind  = ind_list[:iteration+1]
        update   = (v_[ind_mean] + np.sum(v_[sub_ind])) / (2. + iteration)
        next_top = v_[ind_list[iteration+1]]
        if update >= next_top:
            break
    
    v_[ind_mean] = update + 1e-4
    v_[sub_ind]  = update - (1e-4 / (iteration+1.))
    
    return v_
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
    
#%%
