#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module conducts agent-based simulations to evaluate the performance of reinforcement 
learning algorithms with and without reward range normalization across various environmental 
settings. It's specifically designed to test scale-invariant properties of learning models 
by adjusting reward magnitude and discriminability levels.

The simulations proceed by:
1. Setting up environments with specified reward magnitudes and discriminability levels.
2. Launching a designated number of agents to interact within these environments.
3. Recording and storing agents' behavioral data as they navigate through the tasks.
4. Saving the compiled data for subsequent analysis and review.

The aim is to systematically assess how different reward structures affect the decision-making 
efficacies of standard and range-normalized reinforcement learning models.


Created on Thu Feb  6 12:28:23 2025
@author: Maëva L'Hôtellier
"""

# =============================================================================
# IMPORTS
# =============================================================================
import time
import numpy as np
import pandas as pd
import os.path as op
from models import Model
import multiprocessing as mp

# =============================================================================
# SIMULATION-SPECIFIC PARAMETERS
# =============================================================================
decision = 'softmax' # agent's decision policy
n_m = 10 # number of magnitude levels log10-spaced
n_d = 10   # number discriminability levels linearily spaced
n_a = 10 # number of learning rate values linearily spaced. can also be "random" to have one random lr per agent

n_agents = 200
n_trial = 80

# =============================================================================
# INITIALIZE ENVIRONMENT-SPECIFIC PARAMETERS
# =============================================================================
magni  = np.logspace(-6, 6, n_m)
discri = np.linspace(0.1, .99, n_d)

if n_a != "random":
    alpha  = np.linspace(0.1, .99, n_a)
    random_alpha = False
else: 
    alpha = "random"
    random_alpha = True

noise  = [0]
dsigma = [0]


# =============================================================================
# FUNCTIONS TO PERFORM THE SIMULATIONS AND RETURN DATA
# =============================================================================
def launch_simulation(m, d, aExp=None, aCon=None):
    rng = np.random.RandomState()  # Creates a new random state
    rng.randn()
    agent_params= []

    for i in range(n_agents): 
        # Different alphas Qv1 and Qv2 across agent
        alphaQv = rng.rand()
        if aExp == None:
            aExp = rng.rand()
            aCon = rng.rand() * aExp
            assert aExp > aCon > 0

        if decision == "optimistic":
            Qval_init = magni[-1]
            decision2 = 'e-greedy'

        else:
            decision2 = decision
            Qval_init = 0

        agent_params_dict = {'beta_ABS': 1, 
                             'beta_RA': 1,
                             'alphaQ_ABS': alphaQv, 
                             'alphaQ_RA': alphaQv, 
                             'alpha_Qu': None, 
                             'alpha_exp': aExp, 
                             'alpha_con': aCon,     
                             'gamma': 0, 
                             'Qval_init': Qval_init,
                             'decision': decision2}
    
        agent_params.append(agent_params_dict)
    model = Model(n_agents, n_trial, random_learning_rates=False,
                  agent_params=agent_params, talk=False)
    model.set_param_space([m], noise, [d], dsigma, mode="exhaustive")
    model.go()

    return model

def save_data(model, i, j, m, d, k=None, l=None,aExp=None, aCon=None, random_alpha=None):

    # Instrumental accuracy score
    acc_abs = np.mean(model.correct_ABS, axis=(0, 1))
    acc_ra = np.mean(model.correct_RA, axis=(0, 1))
    std_abs = np.std(model.correct_ABS, axis=(0, 1))
    std_ra = np.std(model.correct_RA, axis=(0, 1))
    
    # print(model.correct.shape): (ntrials, ntasks, nagents)
    diff = np.mean(model.correct_RA, axis=0) - np.mean(model.correct_ABS, axis=0) # across trials, per agent
    ra_minus_abs = np.mean(diff) # across agents
    std_ra_minus_abs = np.std(diff)

    assert len(model.Agents_Qval_dist.shape) == 2
    epi_acc_diff = np.mean(model.Agents_Qval_dist - model.Agents_Rm_dist) # the greater the value, the more accurate RA is compared to ABS
    std_epi_acc = np.std(epi_acc_diff)

    Qval_dist = model.Agents_Qval_dist
    assert Qval_dist.shape == (1, n_agents)
    Qval_dist = Qval_dist.flatten()
    std_Qval_dist = np.std(model.Agents_Qval_dist)
    
    Rm_dist = model.Agents_Rm_dist
    assert Rm_dist.shape == (1, n_agents)
    Rm_dist = Rm_dist.flatten()
    std_Rm_dist = np.std(model.Agents_Rm_dist)


    if random_alpha == True:
        data = {
            'ind_magni': i,
            'ind_d': j,
            'val_magni': m,
            'val_discri': d,
            'acc_mean_ABS': acc_abs,
            'acc_std_ABS': std_abs,
            'acc_mean_RA': acc_ra,
            'acc_std_RA': std_ra,
            'ra_minus_abs': ra_minus_abs,
            'std_ra_minus_abs': std_ra_minus_abs,
            'epi_acc_diff': epi_acc_diff,
            'std_epi_acc': std_epi_acc,
            'Qval_dist': Qval_dist,
            'std_Qval_dist': std_Qval_dist,
            'Rm_dist': Rm_dist,
            'std_Rm_dist': std_Rm_dist
            }
        
    if random_alpha == False:
        data = {
            'ind_magni': i,
            'ind_d': j,
            'ind_aExp': k,
            'ind_aCon': l,
            'val_magni': m,
            'val_discri': d,
            'val_aExp': aExp,
            'val_aCon': aCon,
            'acc_mean_ABS': acc_abs,
            'acc_std_ABS': std_abs,
            'acc_mean_RA': acc_ra,
            'acc_std_RA': std_ra,
            'ra_minus_abs': ra_minus_abs,
            'std_ra_minus_abs': std_ra_minus_abs,
            'epi_acc_diff': epi_acc_diff,
            'std_epi_acc': std_epi_acc,
            'Qval_dist': Qval_dist,
            'std_Qval_dist': std_Qval_dist,
            'Rm_dist': Rm_dist,
            'std_Rm_dist': std_Rm_dist
            }
    
    return data


def simulate_and_return_data(p):
    params, random_alpha = p
    if random_alpha:
        i, j, magni, discri = params
        m = magni[i]
        d = discri[j]
        model = launch_simulation(m, d)
        return save_data(model, i=i, j=j, m=m, d=d, random_alpha=True)

    if not random_alpha:
        i, j, k, l, magni, discri, alpha = params
        m = magni[i]
        d = discri[j]
        aExp = alpha[k]
        aCon = alpha[l]
        model = launch_simulation(m, d, aExp, aCon)
        return save_data(model, i=i, j=j, m=m, d=d, k=k, l=l,
                         aExp=aExp, aCon=aCon, random_alpha=False)


def worker_init(df):
    global shared_df
    shared_df = df
    
#%%
# =============================================================================
# PARALLELIZED SIMULATIONS
# =============================================================================


df = pd.DataFrame()

if random_alpha == False:
    parameter_combinations = [(i, j, k, l, magni, discri, alpha)
                              for i in range(len(magni))
                              for j in range(len(discri))
                              for k in range(len(alpha))
                              for l in range(len(alpha))]
if random_alpha == True:
    parameter_combinations = [(i, j, magni, discri)
                          for i in range(len(magni))
                          for j in range(len(discri))]
paired_parameters = [(params, random_alpha) for params in parameter_combinations]

# Define the number of processes to use 
num_processes = mp.cpu_count()

start_time = time.time()

# Get the start method as a string
start_method = mp.get_start_method()

# Initialize a multiprocessing pool with proper initialization and finalization
with mp.get_context(start_method).Pool(num_processes, initializer=worker_init, initargs=(df,)) as pool:
    results = pool.map(simulate_and_return_data, paired_parameters)

# Concatenate the results into the data frame
df = pd.DataFrame(results)
df = df[df['val_aExp'] >= df['val_aCon']]

end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print('Time taken to run:', np.round(execution_time / 60, 3), 'minutes')

start_time2 = time.time()
print('Storing the data...')
df.to_json(op.join(f'/path/to/data/{n_m}m_{n_d}d_{n_a}a_bandits_data.json'), 
                   orient='records')
end_time2 = time.time()
storage_time = end_time2 - start_time2
print('Time taken to store the data:', np.round(storage_time / 60, 3), 'minutes')