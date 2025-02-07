#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to simulate agent performance under various magnitude settings 
and visualize distributions of agent performances across magnitude bins 
for both absolute (ABS) and range-adapted (RA) models.

This script:
    - Sets up model parameters.
    - Runs simulations for a range of magnitude settings.
    - Generates scatter plots, histograms, and performance comparison plots for ABS and RA agents.

Requirements:
    - numpy
    - matplotlib
    - Custom Model class from models.py

Created on Wed Mar 20 20:19:49 2024
@author: Maëva L'Hôtellier
"""

import numpy as np
import matplotlib.pyplot as plt
from models import Model


# =============================================================================
# PARAMETERS
# =============================================================================

MAGNI_SPACE = np.logspace(-2, 2, 50)
NOISE = [0]
DISCRIM = [0.25]
DSIGMA = [0]
N_SUBJECTS = 200
N_TRIALS = 80
BETA_VALUES = [0.5, 1, 5]


# =============================================================================
# SIMULATIONS FUNCTIONS
# =============================================================================


def set_model_params(beta_ABS=1, beta_RA=1):
    """
    Initializes and configures the Model instance with predefined parameters.
    
    Returns:
        Model: An initialized Model instance.
    """
    model = Model(n_subjects=N_SUBJECTS, n_trial=N_TRIALS, 
                  random_learning_rates=True, talk=True,
                  beta_ABS=beta_ABS, beta_RA=beta_RA)
    model.set_param_space(rescale=MAGNI_SPACE, noise=NOISE, discrim=DISCRIM,
                          dsigma=DSIGMA, mode="exhaustive")
    return model


def run_model():
    model = set_model_params()
    model.go()
    return model


def run_simulations_multiple_betas(beta_values):
    """
    Runs the model simulations for a list of beta values.
    
    Args:
        beta_values (list of float): List of beta values to simulate.
    
    Returns:
        list: List of models run for each beta value.
    """
    models = []
    for beta in beta_values:
        model = set_model_params(beta_ABS=beta, beta_RA=beta)
        model.go()
        models.append(model)
    return models


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_individual_performance(correct_matrix, title, color):
    """
    Plots individual performance scatter plot for each magnitude setting.

    Args:
        correct_matrix (ndarray): Array of correct responses for each magnitude.
        title (str): Title for the plot.
        color (str): Color for the scatter plot.
    """
    y_values = correct_matrix.flatten()
    x_values = np.repeat(np.arange(correct_matrix.shape[0]), correct_matrix.shape[1])
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x_values, y_values, color=color, marker='x', s=10)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=.7)
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    plt.xlabel("Magnitude Settings")
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.show()


def plot_histogram(data, title, color):
    """
    Plots histogram of performance for a specific magnitude bin.

    Args:
        data (ndarray): Performance data for the bin.
        title (str): Title for the histogram.
        color (str): Color for the bars in the histogram.
    """
    plt.hist(data, bins=12, color=color, edgecolor='white')
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=.7)
    plt.xlim(-0.1, 1.1)
    plt.title(title)
    plt.xlabel("Performance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_average_performance_per_beta(models, beta_values, title, colors):
    """
    Plots average performance per magnitude per beta setting for given models.
    
    Args:
        models (list of Model): List of models with performance data.
        beta_values (list of float): Corresponding beta values for each model.
        title (str): Title for the plot.
        colors (list of str): List of colors for each beta value line.
    """
    for i, (model, beta, color) in enumerate(zip(models, beta_values, colors)):
        correct_per_m = np.mean(model.correct_ABS if title == "ABS" else model.correct_RA, axis=0)
        avg_correct = np.mean(correct_per_m, axis=1)
        sem_correct = np.std(correct_per_m, axis=1) / np.sqrt(len(correct_per_m))
        
        plt.plot(avg_correct, color=color, alpha=1, label=f"β={beta}")
        plt.fill_between(range(len(MAGNI_SPACE)), avg_correct + sem_correct,
                         avg_correct - sem_correct, alpha=0.3, color=color)
    
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7)
    plt.ylim(0.4, 1)
    plt.title(f"{title} Performance by Magnitude for Various β")
    plt.xlabel("Magnitude")
    plt.ylabel("Average Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":

    # Run model and compute performance
    model = run_model()
    correct_ABS = np.mean(model.correct_ABS, axis=0)
    correct_RA = np.mean(model.correct_RA, axis=0)

    # Plot individual performance scatter plots
    plot_individual_performance(correct_ABS, "Individual ABS Performance per Ma.gnitude", "navy")
    plot_individual_performance(correct_RA, "Individual RA Performance per Magnitude", "firebrick")

    # Plot histograms for performance in low and high magnitude bins
    abs_bin_low_m = correct_ABS[0:10].flatten()
    abs_bin_high_m = correct_ABS[-10:].flatten()
    ran_bin_low_m = correct_RA[0:10].flatten()
    ran_bin_high_m = correct_RA[-10:].flatten()

    plot_histogram(abs_bin_low_m, "ABS Performance (Low Magnitude Bin)", "navy")
    plot_histogram(abs_bin_high_m, "ABS Performance (High Magnitude Bin)", "navy")
    plot_histogram(ran_bin_low_m, "RA Performance (Low Magnitude Bin)", "firebrick")
    plot_histogram(ran_bin_high_m, "RA Performance (High Magnitude Bin)", "firebrick")
    
    # Run simulations for different betas
    models_betas = run_simulations_multiple_betas(BETA_VALUES)
    plot_average_performance_per_beta(models_betas, BETA_VALUES, "ABS", 
                                      ["blue", "navy", "midnightblue"])
    plot_average_performance_per_beta(models_betas, BETA_VALUES, "RA", 
                                      ["salmon", "firebrick", "maroon"])
