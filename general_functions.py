#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides functions to generate rewards, apply noise, compute distances, 
and implement decision-making strategies.

Functions:
    - decision_function: Determines choices based on a specified decision rule.
    - apply_rescale: Scales reward values by a given factor.
    - apply_noise: Adds Gaussian noise to reward values.
    - set_discriminability: Sets initial reward values based on discriminability.
    - random_walk: Applies random variations to rewards, simulating a random walk.
    - create_reward: Generates reward values with discriminability, noise, random walk, and scaling.
    - compute_task_distance: Computes normalized distance between estimated distances and rewards.
    - dist_prop: Computes normalized squared distance between estimate and reward relative to a range.

Dependencies:
    - numpy
    
Created on Thu Dec  9 23:09:01 2021
@author: jeremyperez, Maëva L'Hôtellier
"""

import numpy as np

def decision_function(decision, Q1, Q2, parameters, trial):
    """
    Determines choice between two options based on the specified decision rule.

    Args:
        decision (str): Decision rule to apply ('softmax', 'UCB', or 'e-greedy').
        Q1(float): Q-value for the first option.
        Q2 (float): Q-value for the second option.
        parameters (list): Additional parameters for each decision rule:
                           - softmax: [beta]
                           - UCB: [c, N0, N1, t]
                           - e-greedy: [epsilon]
        trial (int): Current trial number, used to initialize choice randomly if trial == 0.

    Returns:
        int: Chosen option (0 or 1).
    """
    rng = np.random.RandomState() 
    if trial == 0:
        choice = rng.choice([0, 1])
        return choice
    
    if decision == 'softmax':
        beta = parameters[0]
        exp_argument = np.clip(Q1 - Q2, None, 709)  # to avoid runtimewarning Inf exponential value
        Pc = 1/(1 + np.exp(exp_argument * beta))
        choice = int(Pc >= rng.random_sample())
        return choice
    
    elif decision == 'UCB':
        c, N0, N1, t = parameters
        # Sample all options at the beginning of the task
        if N0 == 0:
            return 0
        elif N1 == 0:
            return 1
        

        choice = np.argmax([Q1 + c * np.sqrt(np.log(t)/N0), Q2 + c * np.sqrt(np.log(t)/N1)]) 
        return choice
    
    elif decision == 'e-greedy':
        epsilon = parameters[0]
        exploit = np.argmax([Q1, Q2])
        if epsilon < np.random.random_sample():
            choice = exploit
        else: 
            choice = 1 - exploit
        return choice

    
def apply_rescale(reward, rescale):
    """
    Applies a scaling factor to the reward values.
    """
    return reward * rescale


def apply_noise(reward, noise):
    """
    Adds Gaussian noise to reward values.
    """
    return np.random.normal(reward, noise)


def set_discriminability(d, n_trial):
    """
    Sets reward values for each trial based on a discriminability level.
    
    Args:
        d (float): discriminability level, where higher values increase the difference 
                   between the two reward options. Typical range is 0.1 to 1.
        n_trial (int): Number of trials for which rewards will be generated.
    
    Returns:
        ndarray: An array of shape (n_trial, 2), where each row represents a trial
                 with two options reward values. The options rewards sum to 1.
    """
    
    rewards = np.ones((n_trial, 2))
    rewards[:,1] = 0.5 + d/2
    rewards[:,0] = 0.5 - d/2
    return rewards


def random_walk(rewards, dsigma, seed=10):
    """
    Applies random variations to reward values, simulating a random walk process.
    
    Args:
        rewards (ndarray): Array of initial reward values for each trial and option.
        dsigma (float): Standard deviation of the random walk, controlling variation size.
    
    Returns:
        ndarray: Modified reward array after applying random walk.
    """
    np.random.seed(seed)
    for i in range(1, len(rewards)):
        variation = np.random.normal(0, dsigma, size=(2, 1))
        rewards[i][0] = np.clip(rewards[i - 1][0] + variation[0], 0, 1)
        rewards[i][1] = np.clip(rewards[i - 1][1] + variation[1], 0, 1)
        
    return rewards

def create_reward(rescale, noise, discrim, dsigma, n_trial):
    """
    Generates a reward structure for a series of trials, applying transformations
    for discriminability, random walk, noise, and rescaling.
    
    Args:
        rescale (float): Scaling factor applied uniformly to all rewards.
        noise (float): Standard deviation of Gaussian noise to add to rewards.
        discrim (float): discriminability level to set initial reward values. Higher 
                          values create a larger difference between two options.
        dsigma (float): Standard deviation for the random walk process, adding 
                        variation to rewards across trials.
        n_trial (int): Number of trials for which to generate rewards.
    
    Returns:
        ndarray: A (n_trial, 2) array of reward values, where each row represents 
                 rewards for a trial after applying discriminability, random walk, noise, 
                 and rescaling.
    """
    reward = set_discriminability(discrim, n_trial)
    reward = random_walk(reward, dsigma)
    reward = apply_noise(reward, noise) 
    reward = apply_rescale(reward, rescale)
    return reward


def compute_task_distance(Rm_dist_list, task_reward):
    """
    Computes a normalized task-level distance between estimated distances and actual rewards.
    
    Args:
        Rm_dist_list (list or ndarray): List or array of estimated RA distances for each trial.
        task_reward (list or ndarray): List or array of actual rewards for each trial.
    
    Returns:
        float: Normalized distance metric for the task. This value represents the ratio
               of the sum of estimated distances to the sum of actual rewards, providing
               a task-level comparison of estimated vs. actual reward distances.
    """

    sum_Rm_dist = np.sum(Rm_dist_list)
    sum_rewards = np.sum(task_reward)
    return np.abs(sum_Rm_dist / sum_rewards)


def dist_prop(ev, rew, max_rew, min_rew):
    """
    Computes the normalized squared distance between an estimated value and a reward, 
    relative to the reward range [min_rew, max_rew].

    Args:
        ev (float): Estimated value.
        rew (float): Actual reward, expected to be either max_rew or min_rew.
        max_rew (float): Maximum possible reward value in the range.
        min_rew (float): Minimum possible reward value in the range.

    Returns:
        float: Normalized squared distance between the estimate and the reward.
               If `rew` is `max_rew`, returns the squared proportional error 
               of `ev` relative to `max_rew`. If `rew` is `min_rew`, returns 
               the squared proportional error relative to `min_rew`.

    Raises:
        ValueError: If `rew` is not equal to `max_rew` or `min_rew`.
    """

    if rew not in [max_rew, min_rew]:
        raise ValueError("Reward must be either max_rew or min_rew.")

    if rew == max_rew:
        return ((max_rew - ev) / (max_rew - min_rew)) ** 2

    elif rew == min_rew:
        return ((ev - min_rew) / (max_rew - min_rew)) ** 2
