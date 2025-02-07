#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:32:52 2024

@author: Maëva L'Hôtellier
"""


import ijson
from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt 


#%%
# =============================================================================
# Defining the parameters associated with simulations of interest
# =============================================================================

n_m = 50 # number of iterations for magnitude
n_d = 50 # discriminability
n_a = 20 # learning rate _

fpath =  '/path/to/data'
file = op.join(fpath, f'{n_m}m_{n_d}d_{n_a}a_bandits_data.json')

#%%
# =============================================================================
# Preparing variables to store data of interest
# =============================================================================

m = [] # magnitude level
d = [] # discriminability level
aExp = [] # RA learning rate for range expansion
aCon = [] # RA learning rate for range contraction

acc_mean_ABS = [] # mean accuracy per ABS agent
acc_mean_RA = [] # mean accuracy per RA agent

#%%
# =============================================================================
# Loading the simulated data
# =============================================================================

# Opening the JSON file and reading items one by one
with open(file, 'rb') as fichier:
  
    dictionnaires = ijson.items(fichier, 'item')
    for dictionnaire in tqdm(dictionnaires, desc="Processing items"): # total items: 475000it
        m.append(float(dictionnaire['val_magni']))
        d.append(float(dictionnaire['val_discri']))
        array_acc_mean_ABS = np.array([float(x) for x in dictionnaire['acc_mean_ABS']])
        array_acc_mean_RA  = np.array([float(x) for x in dictionnaire['acc_mean_RA']]) 
        aExp.append(float(dictionnaire['val_aExp']))
        aCon.append(float(dictionnaire['val_aCon']))
    
        acc_mean_ABS.append(array_acc_mean_ABS)
        acc_mean_RA.append(array_acc_mean_RA)

# Convert lists of arrays to lists of lists
acc_mean_abs_lists = [list(arr) for arr in acc_mean_ABS]
acc_mean_ra_lists = [list(arr) for arr in acc_mean_RA]

# Create the DataFrame
df = pd.DataFrame({
    'val_magni': m,
    'val_discri': d,
    'aExp': aExp,
    'aCon': aCon,
    'acc_mean_ABS': acc_mean_abs_lists,
    'acc_mean_RA': acc_mean_ra_lists
})


#%%
# =============================================================================
# Functions
# =============================================================================


def elementwise_mean(series):
    """
    Calculate the element-wise mean of a series of lists.

    This function takes a list of lists (`series`), where each sublist must be of the same length,
    and computes the mean of each element across all lists. The result is a list where each position
    corresponds to the mean of the elements at that position in the input lists.

    Parameters:
    - series (list of lists of float): A list where each element is a list of numbers. All sublists must
      have the same number of elements.

    Returns:
    - numpy.ndarray: A 1D array where each element is the mean of the elements at that position across
      all input lists.
    """
    list_lengths = [len(x) for x in series]
    if len(set(list_lengths)) != 1: # Ensure all lists are the same length
        raise ValueError("All lists must be of the same length.")
    
    return np.mean(np.array([np.array(lst) for lst in series]), axis=0)

# After grouping the data by magnitude level and computing element-wise means 
# using the defined function, extract the mean and standard deviation for both
# ABS and RA accuracy metrics from the grouped data to plot their central tendency per magnitude level. 
def plot_performance(grouped_mean):
    score_abs = []
    sd_abs = []
    score_ra = []
    sd_ra = []
    magni = []
    
    for i, row in grouped_mean.iterrows():
        score_abs.append(np.mean(row['acc_mean_ABS']))
        sd_abs.append(np.std(row['acc_mean_ABS']))
        score_ra.append(np.mean(row['acc_mean_RA']))
        sd_ra.append(np.std(row['acc_mean_RA']))
        magni.append(row.name)
    
    
    # Convert lists to numpy arrays for element-wise operations
    score_abs_array = np.array(score_abs)
    sd_abs_array = np.array(sd_abs)
    score_ra_array = np.array(score_ra)
    sd_ra_array = np.array(sd_ra)
    
    # Calculate bounds for the fill_between
    upper_bound_abs = score_abs_array + sd_abs_array
    lower_bound_abs = score_abs_array - sd_abs_array
    upper_bound_ra = score_ra_array + sd_ra_array
    lower_bound_ra = score_ra_array - sd_ra_array
    
    # Plotting the performance metrics
    plt.figure(figsize=(5,5))
    # Plotting ABS
    plt.plot(score_abs_array, color='navy', label='ABS')
    plt.fill_between(range(len(score_abs_array)), upper_bound_abs,
                     lower_bound_abs, color='navy', alpha=0.3)
    
    # Plotting RA
    plt.plot(score_ra_array, color="firebrick", label='RA')
    plt.fill_between(range(len(score_ra_array)), upper_bound_ra,
                     lower_bound_ra, color='firebrick', alpha=0.3)
    
    # Add details
    varname = grouped_mean.index.name
    plt.xlabel(varname)
    if 'magni' in varname:
        varvals = np.unique(m)
        var_len = n_m
        x_labels = np.round(np.log10(varvals[range(0, var_len, 10)].tolist() + [varvals[var_len - 1]]))

    elif 'discri' in varname:
        varvals = np.unique(d)
        var_len = n_d
        x_labels = np.round(varvals[range(0, var_len, 10)].tolist() + [varvals[var_len - 1]], 2)

    plt.xticks(list(range(0, var_len, 10)) + [var_len - 1], labels=x_labels)
    plt.axhline(0.5, ls='--', c='k', alpha=0.2)
    plt.ylabel('Proportion Correct')
    plt.ylim(0.2, 1.0)
    plt.legend()
    plt.show()


#%%
# =============================================================================
# Figure 4A
# =============================================================================


# Compute average performance across magnitude levels,
# learning rates, and across agents for each discriminability level


grouped_mean_discri = df.groupby('val_discri').agg({
    'val_magni': 'mean',
    'aExp': 'mean',
    'aCon': 'mean',
    'acc_mean_ABS': elementwise_mean,
    'acc_mean_RA': elementwise_mean
})

plot_performance(grouped_mean_discri)

#%%
# =============================================================================
# Figure 4C
# =============================================================================


# Compute average performance across discriminability levels,
# learning rates, and across agents for each magnitude level

grouped_mean_magni = df.groupby('val_magni').agg({
    'val_discri': 'mean',
    'aExp': 'mean',
    'aCon': 'mean',
    'acc_mean_ABS': elementwise_mean,
    'acc_mean_RA': elementwise_mean
})

plot_performance(grouped_mean_magni)
