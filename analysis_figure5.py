#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script plots results across stationary, shrinking, expanding, and random
walk reward environments.

Created on Wed Oct 30 18:08:09 2024
@author: Maëva L'Hôtellier
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import sem
from agents import Agent
from general_functions import set_discriminability, random_walk


# =============================================================================
# PARAMETERS
# =============================================================================
n_subjects = 1000
n_tasks = 1
seed = 42
np.random.seed(seed)

# Define main parameters (dict_params)
dict_params = {
    'beta_ABS': 1, 'beta_RA': 1,
    'alphaQ_ABS': 0.58, 'alphaQ_RA': 0.58,
    'alpha_exp': 0.75, 'alpha_con': 0.056,
    'alpha_Qu': None, 'gamma': None,
    'Qval_init': 0,  'decision': 'softmax'}

# =============================================================================
# SIMULATIONS FUNCTIONS
# =============================================================================

def get_models_performance(rewards, n_trial, dict_params, n_subjects, n_tasks, random_lr=False):

    """
    Compute performance of models based on specified rewards and parameters.
    Returns a dictionary of performance data for further analysis and plotting.
    """

    # Initialize storage arrays
    shape_3d = (n_trial, n_tasks, n_subjects)
    data = {
        'correct_ABS': np.zeros(shape_3d),
        'Qv1_ABS': np.zeros(shape_3d),
        'Qv2_ABS': np.zeros(shape_3d),
        'correct_RA': np.zeros(shape_3d),
        'Qv1_RA': np.zeros(shape_3d),
        'Qv2_RA': np.zeros(shape_3d),
        'Rmin': np.zeros(shape_3d),
        'Rmax': np.zeros(shape_3d)}

    for i in range(n_subjects):
        agent = Agent(**dict_params)
        
        # Have the agent perform the task
        agent.perform_task(rewards)

        # Store the results for this agent
        data['correct_ABS'][:, 0, i] = agent.correct_ABS
        data['Qv1_ABS'][:, 0, i] = agent.Qv1_ABS[:-1]
        data['Qv2_ABS'][:, 0, i] = agent.Qv2_ABS[:-1]

        data['correct_RA'][:, 0, i] = agent.correct_RA
        data['Qv1_RA'][:, 0, i] = agent.Qv1_RA[:-1]
        data['Qv2_RA'][:, 0, i] = agent.Qv2_RA[:-1]
        data['Rmin'][:, 0, i] = agent.Rmin[:-1]
        data['Rmax'][:, 0, i] = agent.Rmax[:-1]
    return data


def setup_rewards_stationary():
    rewards1 = [0.6] * 80
    rewards2 = 1 - np.array(rewards1)
    return np.column_stack((rewards2, rewards1))


def setup_rewards_expanding():
    rewards1 = [0.6] * 20 + [0.7] * 20 + [0.8] * 20 + [0.9] * 20
    rewards2 = 1 - np.array(rewards1)
    rewards = np.column_stack((rewards2, rewards1))  # Combine into a 2D array for both options
    return rewards


def setup_rewards_shrinking():
    rewards1 = [0.9] * 20 + [0.8] * 20 + [0.7] * 20 + [0.6] * 20
    rewards2 = 1 - np.array(rewards1)
    return np.column_stack((rewards2, rewards1))


def setup_rewards_drifting(dsigma):
    rewards = set_discriminability(d=0.5, n_trial=80)
    return random_walk(rewards, dsigma=dsigma)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_rewards_trajectory(rewards):
    """
    Plot the trajectory of rewards over trials for visualization.
    """
    sns.set(style="whitegrid", font_scale=3)
    plt.figure(figsize=(4, 5))
    plt.plot(rewards[:, 0], label="Reward 1", linewidth=2.2)
    plt.plot(rewards[:, 1], label="Reward 2", linewidth=2.2)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7)

    plt.xlabel("Trials")
    plt.ylabel("Reward Values")
    plt.show()



def plot_latent(rewards, n_trial, results):
    sns.reset_orig()

    # Calculate means and standard errors
    rmax = np.mean(results['Rmax'], axis=(1, 2))
    rmin = np.mean(results['Rmin'], axis=(1, 2))
    sem_rmax = sem(results['Rmax'], axis=2, nan_policy='raise').ravel()
    sem_rmin = sem(results['Rmin'], axis=2, nan_policy='raise').ravel()
  
    
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

    # Plot real rewards
    sns.lineplot(x=range(n_trial), y=rewards[:, 1],
                  ax=ax, color='mediumpurple', linewidth=2.5, label="option A")
    sns.lineplot(x=range(n_trial), y=rewards[:, 0],
                  ax=ax, color='gold', linewidth=2.5, label="option B")

    # Plot Rmax with shaded error region
    sns.lineplot(x=range(n_trial), y=rmax, ax=ax,
                  color='k', linewidth=2, label='Rmax')
    ax.fill_between(range(n_trial), rmax - sem_rmax, rmax + sem_rmax,
                    alpha=0.2, color='k')

    sns.lineplot(x=range(n_trial), y=rmin, ax=ax,
                  color='k', linewidth=2, linestyle = '--', label='Rmin')
    ax.fill_between(range(n_trial), rmin - sem_rmin, rmin + sem_rmin,
                    alpha=0.2, color='k')

    correct_ABS = np.mean(results['correct_ABS'], axis=2)
    correct_RA = np.mean(results['correct_RA'], axis=2)
    acc_diff = correct_RA - correct_ABS

    i = 0
    for t, acc in  enumerate(acc_diff):

         if acc > 0:
             c = "firebrick"
             if i == 0:
                 plt.scatter(t, 1.05, color=c, edgecolor='w', alpha=.7, label='RA > ABS')
                 i += 1
             else:
                 plt.scatter(t, 1.05, color=c, edgecolor='w', alpha=.7)

    plt.ylim(0, 1.1)
    plt.grid(False)
    ax.tick_params(color='k', labelcolor='k')
    for spine in ax.spines.values():
        spine.set_edgecolor('k')
        plt.legend(loc='best')
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Define all reward environments
    reward_envs = {
        "stationary": setup_rewards_stationary(),
        "shrinking": setup_rewards_shrinking(),
        "expanding": setup_rewards_expanding(),
        "Slowly drifting": setup_rewards_drifting(dsigma=0.05),
        "Quickly drifting": setup_rewards_drifting(dsigma=0.1),
    }

    # Process each environment
    for env_name, rewards in reward_envs.items():
        print(f"Processing {env_name} environment...")

        # Perform analysis
        n_trial = rewards.shape[0]
        results = get_models_performance(rewards, n_trial, dict_params, 
                                         n_subjects, n_tasks, random_lr=False)
        # Plot
        plot_latent(rewards, n_trial, results)