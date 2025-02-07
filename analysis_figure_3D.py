#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:06:51 2024

@author: Maëva L'Hôtellier
"""

import time
import numpy as np
import progressbar
import seaborn as sns
import hyperopt as hpo
import matplotlib.pyplot as plt
from models import Model

# =============================================================================
# PARAMETERS
# =============================================================================

REWARDS = [0.25, 0.75]
BETA_VALUES = np.linspace(0.1, 100, 500)
MAGNI_VALUES = np.logspace(-1, 1, 500)
ALPHA_Q = 0.3
C_FACTOR = 3
ALPHA_CONDS = {'alpha': 'set_alpha',
                'alphaQ_ABS' : 0.3, 
                'alphaQ_RA': 0.3,
                'alpha_exp': 0.28,
                'alpha_con': 0.22 
              }

# =============================================================================
# TRADE-OFF QUALITY FUNCTION
# =============================================================================

def measure_tradeoff_quality(beta, rewards, magni, alpha_Q, C):
    """
    Computes the quality of trade-offs based on beta, magnitude, and reward settings.
    
    Args:
        beta (ndarray): Array of beta values.
        rewards (list): List of two rewards representing decision outcomes.
        magni (ndarray): Array of magnitude values.
        alpha_Q_ABS (float): learning rate for Q-values update.
        C (float): Scaling factor.

    Returns:
        ndarray: Quality matrix with dimensions (len(beta), len(magni)).
    """

    quality = np.zeros((len(beta), len(magni)))
    for i, b in enumerate(beta):
        for j, m in enumerate(magni):
            p_a_to_b = 1 / (1 + np.exp(b * alpha_Q * m * rewards[0]))
            p_b_to_a = 1 / (1 + np.exp(b * m * (rewards[1] - rewards[0])))
            quality[i, j] = C * (p_a_to_b - p_b_to_a)
    return quality

# =============================================================================
# SEARCHING BETA APPARENT FOR THE RANGE MODEL
# =============================================================================

def loss_Qval_distance(model):
    """    
    This function calculates the beta that minimizes the difference between
    (Q1_RA - Q2_RA) and (beta * (Q1_ABS - Q2_ABS) across trials
    and agents for per magnitude value.
    
    Args:
        model (Model): The model instance containing Q-value differences.
        
    Returns:
        float: Sum of squared differences for the beta fitting loss.
    """
    
    # Calculate the difference array between the RA and the ABS beta-scaled Q-values
    diff_arr = model.deltaQ_RA - model.Beta_deltaQ_ABS # Trial-wise
    
    # Square the differences to penalize larger deviations
    squared_diffs = diff_arr ** 2
    
    # Sum the squared differences across trials and agents
    sum_diffs = np.sum(squared_diffs, axis=(0, 2))

    return sum_diffs


def run_minimization(magni, alpha_conds, min_beta, max_beta, max_evals=3000, talk=False):
    """
    Runs hyperparameter optimization to find the optimal beta for each magnitude value.
    Optimization minimizes the loss function to best match Q-value differences in RA and ABS models.
    
    Args:
        magni (list): List of magnitudes for which to optimize beta.
        alpha_conds (dict): Condition settings for alpha values, including:
                            - 'alpha': Type of learning rate ('random' or 'set_alpha').
                            - If 'set_alpha', should also include specific values 
                              for 'alphaQ_ABS', 'alpha_Q_RA', etc.
        min_beta (float): Minimum beta value for the search space.
        max_beta (float): Maximum beta value for the search space.
        max_evals (int): Maximum evaluations for the optimization process.
        talk (bool): If True, enables verbose output for progress monitoring.
    
    Returns:
        np.ndarray: Array of optimal beta values for each magnitude in `magni`.
    """
    def minimize(m):
        # Define the search space for beta
        space = {'beta': hpo.hp.uniform('beta', min_beta, max_beta)}
    
        def agents_journey(beta):
          # Initialize the Model instance based on `alpha_conds`
          n_agents = 1
          # Case of random learning rates
          if alpha_conds['alpha'] == 'random':
              model = Model(n_subjects=n_agents, n_trial=80, 
                            random_learning_rates=True, 
                            beta_ABS=beta, beta_RA=1, talk=False)
    
          # Case of pre_determined learning rates
          elif alpha_conds['alpha'] == 'set_alpha':
              required_keys = {'alphaQ_ABS', 'alphaQ_RA', 'alpha_exp', 'alpha_con'}
              if not required_keys.issubset(alpha_conds):
                  raise ValueError("Missing required alpha condition keys in `alpha_conds`.")

              # Define agent parameters
              agent_params = {
                  'beta_ABS': beta, 'beta_RA': 1,
                  'alphaQ_ABS': alpha_conds['alphaQ_ABS'], 
                  'alphaQ_RA': alpha_conds['alphaQ_RA'],
                  'alpha_Qu': None,
                  'alpha_exp': alpha_conds['alpha_exp'], 
                  'alpha_con': alpha_conds['alpha_con'], 
                  'gamma': None,
                  'Qval_init': 0,
                  'decision': 'softmax', 
                  'eps_ABS': None, 'eps_RA': None, 
                  'c_ABS': None, 'c_RA': None
                  } 
    
              # Initialize model with same parameters for each agent
              agent_params_list = [agent_params for _ in range(n_agents)]
              model = Model(n_subjects=n_agents, n_trial=80, 
                            random_learning_rates=False, 
                            agent_params=agent_params_list, talk=False)
      
          # Set model parameters and execute
          model.set_param_space(rescale=[m], noise=[0], discrim=[0.5], # discrim of 0.5 makes rewards = [0.25, 0.75] * m 
                                dsigma=[0], mode="exhaustive")
          model.go()
              
          # Calculate and return the loss
          return loss_Qval_distance(model)[0]

        # Wrapper for optimization process
        def f(params): 
            return agents_journey(**params)
        
        # Run optimization for the current magnitude `m`
        verbose = True if talk else False
        best_beta = hpo.fmin(
            fn=f,
            space=space,
            algo=hpo.tpe.suggest,
            max_evals=max_evals,
            verbose=verbose
        )
        return best_beta

    # Init array to store opti betas and loss for each magnitude
    apparent_beta = np.zeros((len(magni), 1))

    # Visualize progress
    bar = progressbar.ProgressBar(maxval=len(magni), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start_time = time.time()

    for i, m in enumerate(magni):
        print('-------')
        print(f'Iteration {i}/{len(magni)}')
        print(f"Finding the apparent beta for magnitude {m}...")
        bar.update(i+1)
        best_beta = minimize(m)
        apparent_beta[i] = best_beta['beta']
        print(f'In magnitude {m} apparent beta found={best_beta["beta"]}')
        
    end_time = time.time()
    elapsed_time = end_time - bar.start_time
    elapsed_minutes = elapsed_time / 60
    bar.finish()
    print(f"Time taken to run: {elapsed_minutes:.2f} minutes")

    return apparent_beta

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_matrix(quality, magnis, betas, apparent_beta=None):
    """
    Plots a heatmap of the quality matrix with custom x and y labels, and an optional overlay 
    for `apparent_beta` estimates.
    
    Args:
        quality (ndarray): Quality matrix to plot.
        magnis (ndarray): Array of magnitude values.
        betas (ndarray): Array of beta values.
        apparent_beta (ndarray, optional): Optional array of best beta estimates to overlay on the plot.
    """
    fig, ax = plt.subplots()
    sns.heatmap(quality, ax=ax, cmap='inferno', cbar_kws={'label': 'Quality'},
                square=True, xticklabels=False, yticklabels=False)
    ax.invert_yaxis()

    # Define x-axis ticks based on magnitude labels
    tgt_xlabels = np.logspace(np.log10(magnis[0]), np.log10(magnis[-1]), num=10)
    x_locs = [np.argmin(np.abs(magnis - tgt)) for tgt in tgt_xlabels]
    x_vals = [f"{val:.2g}" if val < 0.01 else f"{val:.2f}" for val in tgt_xlabels]
    ax.set_xticks(x_locs)
    ax.set_xticklabels(x_vals, rotation=45)

    # Define y-axis ticks based on beta labels
    y_vals = np.linspace(betas.min(), betas.max(), 10)
    y_locs = [np.argmin(np.abs(betas - y_val)) for y_val in y_vals]
    y_locs = [idx + 0.5 for idx in y_locs] # center on each cell
    ax.set_yticks(y_locs)
    ax.set_yticklabels([f"{y_val:.1f}" for y_val in y_vals])

    ax.set_xlabel("Magnitude Value")
    ax.set_ylabel("ß Value")
    
    if apparent_beta is not None:
        # Find the closest index in `betas` for each value in `apparent_beta`
        # 0.5 is a constant that ensures the dot will be placed in the middle of the cell
        beta_indices = [np.argmin(np.abs(betas - beta)) + 0.5 for beta in apparent_beta]
        # Overlay the `apparent_beta` as scatter points at the calculated index positions
        ax.scatter(np.arange(len(apparent_beta)) + 0.5, # center on each cell
                   beta_indices, s=10, color='w', edgecolor='firebrick')
        
        opti_beta_idx = np.argmax(quality_matrix, axis=0) + 0.5
        ax.plot(np.arange(len(apparent_beta)), opti_beta_idx, c='navy')
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":

    # Measure trade-off quality and plot
    quality_matrix = measure_tradeoff_quality(BETA_VALUES, REWARDS, 
                                              MAGNI_VALUES, ALPHA_Q, C_FACTOR)
    plot_matrix(quality_matrix, MAGNI_VALUES, BETA_VALUES)

    max_indices = np.argmax(quality_matrix, axis=0)

    apparent_beta = run_minimization(MAGNI_VALUES, ALPHA_CONDS, min_beta=BETA_VALUES.min(), 
                 max_beta=BETA_VALUES.max(), max_evals=3000, talk=True)
    plot_matrix(quality_matrix, MAGNI_VALUES, BETA_VALUES, apparent_beta)
