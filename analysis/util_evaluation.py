#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:58:58 2019

@author: zhang64
"""


import torch
import numpy as np
import torch.nn.parallel

from KDEpy import FFTKDE

#%%
def mirror_1d(d, 
              xmin=None, 
              xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), 
                               d, 
                               (2*xmax-d[d >= xmed]).reshape(-1,1)))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, 
                               d))
    elif xmax is not None:
        return np.concatenate((d, 
                               2*xmax-d))
    else:
        return d

#%%
def ece_kde_binary(p,
                   label,
                   p_int=None,
                   order=1,
                   top_p=1):

    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p,1e-256,1-1e-256)
    p_int = np.clip(p_int,1e-256,1-1e-256)
        
    x_int = np.linspace(-0.6, 1.6, num=2**14)    
    
    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    label_index = np.array([np.where(r==1)[0][0] for r in label]) # 0.9901 # default: 1 # max(r)
    with torch.no_grad():
        if p.shape[1] !=2:
            p_new = torch.from_numpy(p)
            p_b = torch.zeros(N,1) # winning score
            label_binary = np.zeros((N,1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
        else:
            p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index
                
    method = 'triweight'
    
    dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
    kbw = np.std(p_b.numpy())*(N*2)**-0.2
    kbw = np.std(dconf_1)*(N*2)**-0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1
    
    
    p_int = p_int/np.sum(p_int,1)[:,None]
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1,1))
        if p_int.shape[1]!=2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i,pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i,1]

    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    
    if p.shape[1] !=2: # top label (confidence)
        perc = np.mean(label_binary)
    else: # or joint calibration for binary cases
        perc = np.mean(label_index)
            
    integral = np.zeros(x_int.shape)
    reliability= np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]
        if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu)==False:
                integral[i] = np.abs(conf-accu)**order*pp2[i]  
                reliability[i] = accu
        else:
            if i>1:
                integral[i] = integral[i-1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
#%%
def ece_kde_binary_dev(p,
                       label,
                       p_int=None,
                       order=1,
                       top_p=1):
    '''
    version dev
    '''
    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p,1e-256,1-1e-256)
    p_int = np.clip(p_int,1e-256,1-1e-256)
        
    x_int = np.linspace(-0.6, 1.6, num=2**14) # x points
    
    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    label_index = np.array([np.where(r==1)[0][0] for r in label]) # 0.9901 # default: 1 # max(r)
    with torch.no_grad():
        if p.shape[1] !=2:
            p_new = torch.from_numpy(p)
            p_b = torch.zeros(N,1) # winning score
            label_binary = np.zeros((N,1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
        else:
            p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index
                
    method = 'triweight'
    
    dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy() # the correct winning scores
    kbw = np.std(p_b.numpy())*(N*2)**-0.2
    kbw = np.std(dconf_1)*(N*2)**-0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1, low_bound, up_bound) # considering two boundaries
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1
    # now we have: density distribution of the *correct* winning scores 
    
    # ~~~~~~~~~~ ~~~~~~~~~~    
    p_int = p_int/np.sum(p_int,1)[:,None] # normalization (not softmax, no exp)
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1,1))
        if p_int.shape[1]!=2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i,pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i,1]
    # I think it's the same as p_b

    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1
    # now we have: density distribution of the *all* winning scores 
    
    '''
    plt.plot(pp1, label='correct winning scores')
    plt.plot(pp2, label='all winning scores')
    plt.legend()
    '''
    
    # ~~~~~~~~~~ ~~~~~~~~~~    
    if p.shape[1] !=2: # top label (confidence)
        perc = np.mean(label_binary)
    else: # or joint calibration for binary cases
        perc = np.mean(label_index)
    # it's the same as accu
        
    # ~~~~~~~~~~ ~~~~~~~~~~    
    integral = np.zeros(x_int.shape)
    reliability= np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]
        here = np.abs(x_int-conf).argmin()
        if np.max([pp1[here], pp2[here]]) > 1e-6:
            accu = np.min([perc*pp1[here]/pp2[here], # Note: perc here
                           1.0])
            '''
            if perc*pp1[here]/pp2[here] > 1:
                print(perc*pp1[here]/pp2[here])
            '''
            if np.isnan(accu)==False:
                integral[i] = np.abs(conf-accu)**order*pp2[i]  
                reliability[i] = accu
        else:
            if i>1:
                integral[i] = integral[i-1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind], x_int[ind]) / np.trapz(pp2[ind], x_int[ind]), perc

#%%
def ece_hist_binary(p, 
                    label, 
                    n_bins=15, 
                    order=1,
                    printout=False):
    
    p = np.clip(p,1e-256,1-1e-256)
    
    N = p.shape[0] # how many samples
    label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index
    with torch.no_grad():
        if p.shape[1] !=2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N): # iter over samples
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i,pred_label]/torch.sum(preds_new[i,:]) # sum to 1
        else:
            preds_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary) # means hit or not


        x = confidences.numpy()
        x = np.sort(x,axis=0)
        binCount = int(len(x)/n_bins) #number of data points in each bin
        bins = np.zeros(n_bins) #initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i+1) * binCount,x.shape[0]-1)]
            #print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins)+1,1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1,1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        
        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean() # means the percentage of this bin
            #print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin)**order * prop_in_bin
                if printout:
                    print('{:.3f} - {:.3f}: confi {:.3f}, accu {:.3f}, dif {:.3f}'.format(
                                                    bin_lower.numpy()[0],
                                                    bin_upper.numpy()[0],
                                                    avg_confidence_in_bin.numpy(),
                                                    accuracy_in_bin.numpy(),
                                                    avg_confidence_in_bin.numpy() - accuracy_in_bin.numpy()
                                                    ))
    return ece_avg
#%%
def ece_hist_binary2(p, 
                     label, 
                     lo_s=[], # horse
                     n_bins=15, 
                     order=1,
                     printout=False):
    '''
    here we use confidence and confidence2
    version 2: understand it
    '''
    p = np.clip(p,1e-256,1-1e-256)
    
    N = p.shape[0] # how many samples
    label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index
    with torch.no_grad():
        if p.shape[1] !=2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N): # iter over samples
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i,pred_label]/torch.sum(preds_new[i,:]) # sum to 1
        else:
            preds_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index

        # ~~~~~~~~~~ default
        confidences = preds_b
        # ~~~~~~~~~~ default
        if 0 == len(lo_s):
            lo_s = np.zeros_like(preds_b)
        elif len(lo_s.shape) == 1:
            lo_s = lo_s[:,None]
        confidences2 = preds_b - torch.from_numpy(lo_s)
        # print(confidences)
        accuracies = torch.from_numpy(label_binary) # means hit or not


        x = confidences.numpy()
        x = np.sort(x,axis=0)
        binCount = int(len(x)/n_bins) #number of data points in each bin
        bins = np.zeros(n_bins) #initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i+1) * binCount,x.shape[0]-1)]
            #print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins)+1,1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1,1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        
        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean() # means the percentage of this bin
            #print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences2[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin)**order * prop_in_bin
                if printout:
                    print('{:.3f} - {:.3f}: confi {:.3f}, accu {:.3f}, dif {:.3f}'.format(
                                                    bin_lower.numpy()[0],
                                                    bin_upper.numpy()[0],
                                                    avg_confidence_in_bin.numpy(),
                                                    accuracy_in_bin.numpy(),
                                                    avg_confidence_in_bin.numpy() - accuracy_in_bin.numpy()
                                                    ))
    return ece_avg
#%%
def ece_hist_binary_dev(p, 
                        label, 
                        loss=[], # horse
                        n_bins=15, 
                        order=1,
                        printout=False,
                        bin_lowers=[],
                        bin_uppers=[]):
    '''
    here we use confidence and confidence2
    version dev
    
    testing:
    p = logit
    label = label
    loss = []
    bin_lowers = []
    bin_uppers = []
    '''
    p = np.clip(p,1e-256,1-1e-256)
    
    N = p.shape[0] # how many samples
    label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index
    with torch.no_grad():
        if p.shape[1] !=2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N): # iter over samples
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i,pred_label]/torch.sum(preds_new[i,:]) # sum to 1
        else:
            preds_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index

        # default ~~~~~~~~~~
        confidences = preds_b
        # default ~~~~~~~~~~
        if 0 == len(loss):
            print('> using original confidence.')
            loss = np.zeros_like(preds_b)
        elif len(loss.shape) == 1:
            print('> adjust confidence.')
            loss = loss[:,None]
        # new confidence:
        confidences2 = preds_b - torch.from_numpy(loss)
        # print(confidences)
        accuracies = torch.from_numpy(label_binary) # means hit or not
        
        # get the bins
        if len(bin_lowers) == 0 and len(bin_uppers) == 0:
            x = confidences.numpy()
            x = np.sort(x,axis=0)
            binCount = int(len(x)/n_bins) #number of data points in each bin
            bins = np.zeros(n_bins) #initialize the bins values
            for i in range(0, n_bins, 1):
                bins[i] = x[min((i+1) * binCount,x.shape[0]-1)]
                #print((i+1) * binCount)
            bin_boundaries = torch.zeros(len(bins)+1,1)
            bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1,1)
            bin_boundaries[0] = 0.0
            bin_boundaries[-1] = 1.0
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            print('> using calculated bins.')
            BIN_DEFINED = False
        else:
            print('> using defined bins.')
            BIN_DEFINED = True
        
        # now we can calculate the integral of differences
        ece_avg = torch.zeros(1)
        ibin    = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences2.gt(bin_lower.item()) * confidences2.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean() # means the percentage of this bin
            #print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                # count using new confidence
                avg_confidence_in_bin = confidences2[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin)**order * prop_in_bin
                if printout:
                    print('{:02}({:03})> {:.3f} - {:.3f}: confi {:.3f}, accu {:.3f}, dif {:.3f}'.format(
                                                    ibin, sum(in_bin.numpy())[0],
                                                    bin_lower.numpy()[0],
                                                    bin_upper.numpy()[0],
                                                    avg_confidence_in_bin.numpy(),
                                                    accuracy_in_bin.numpy(),
                                                    avg_confidence_in_bin.numpy() - accuracy_in_bin.numpy()
                                                    ))
                    ibin += 1
    if BIN_DEFINED:
        return ece_avg.numpy()[0]
    else:
        return ece_avg.numpy()[0], bin_lowers, bin_uppers, label_binary
#%%
def ece_eval_binary(p, 
                    label,
                    top_p=1):
    mse = np.mean(np.sum((p-label)**2,1)) # Mean Square Error
    N = p.shape[0]
    nll = -np.sum(label*np.log(p))/N # log_likelihood
    accu = (np.sum((np.argmax(p,1)-np.array([np.where(r==1)[0][0] for r in label]))==0)/p.shape[0]) # Accuracy # 0.9901 # default: 1 # max(r)
    
    # if hist is used
    ## ece = ece_hist_binary(p,label).cpu().numpy() # ECE
    # or if KDE is used
    ece = ece_kde_binary(p,
                         label,
                         top_p=top_p)

    return ece, nll, mse, accu
#%%
def ece_eval_binary_hist(p, 
                         label,
                         top_p=1,
                         bin_lowers=[],
                         bin_uppers=[]):
    mse = np.mean(np.sum((p-label)**2,1)) # Mean Square Error
    N = p.shape[0]
    nll = -np.sum(label*np.log(p))/N # log_likelihood
    accu = (np.sum((np.argmax(p,1)-np.array([np.where(r==1)[0][0] for r in label]))==0)/p.shape[0]) # Accuracy # 0.9901 # default: 1 # max(r)
    
    # if hist is used
    ## ece = ece_hist_binary(p,label).cpu().numpy() # ECE
    # or if KDE is used
    '''
    ece = ece_kde_binary(p,
                         label,
                         top_p=top_p)
    '''
    ece = ece_hist_binary_dev(p, label, # our aim data
                              loss=[], printout=False,
                              bin_lowers=bin_lowers, bin_uppers=bin_uppers)  
    return ece, nll, mse, accu
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