#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the `Agent` class, representing an individual decision-making agent 
in a simulated environment. The agent performs tasks using either an absolute(ABS) or a range-adapted (RA) 
decision-making model and updates its beliefs based on received rewards and the chosen model.

Class:
    - Agent: Represents an agent that learns and makes decisions across trials using 
             customizable decision rules (softmax, epsilon-greedy, UCB, policy-based).
             
Key Methods:
    - perform_task: Executes a series of trials within a task, tracking rewards and performance.
    - perform_trial: Runs a single trial within a task, applying the chosen model.
    - reinit_results: Resets result storage to prepare the agent for a new task.

Dependencies:
    - numpy
    - general_functions

Created on Thu Dec  9 23:10:46 2021

@author: jeremyperez, Maëva L'Hôtellier
"""


import numpy as np
from general_functions import dist_prop, decision_function, compute_task_distance

class Agent:

    """
    Represents an agent that makes decisions and learns over a series of trials using 
    either ABS or RA learning. The agent's behavior is controlled by 
    parameters that influence decision-making and learning updates.
    
    Attributes:
        n_trial (int): Number of trials per task.
        decision (str): The type of decision function ('softmax', 'e-greedy', 'UCB', or 'policy').
        ...
    """

    def __init__(self, beta_ABS, beta_RA, alphaQ_ABS, alphaQ_RA, alpha_Qu, alpha_exp,
                 alpha_con, gamma, Qval_init=0, decision='softmax', eps_ABS=0.05, 
                 eps_RA=0.05, c_ABS=1, c_RA=1, n_trial=80):
        """
        Initializes an agent with specific decision-making and learning parameters.
        
        Args:
            beta_ABS (float): Softmax temperature for ABS model.
            beta_RA (float): Softmax temperature for RA model.
            alphaQ_ABS (float): Learning rate for Q-values in ABS model.
            alphaQ_RA (float): Learning rate for Q-values in RA model.
            alpha_Qu (float): Learning rate for unchosen option.
            alpha_exp (float): Learning rate for RA range expansion.
            alpha_con (float): Learning rate for RA range contraction.
            gamma (float): Discount factor applied to future rewards.
            Qval_init (float, optional): Initial Q-value for decision options. 
            decision (str, optional): Decision rule, can be 'softmax', 'UCB', or 'e-greedy'. 
            eps_ABS (float, optional): Epsilon value for exploration in ABS model. 
            eps_RA (float, optional): Epsilon value for exploration in RA model. 
            c_ABS (float, optional): 'c' parameter for UCB policy in ABS model. 
            c_RA (float, optional): 'c' parameter for UCB policy in RA model.
            n_trial (int, optional): Number of trials per task.
        """
        # --- Decision parameters ---
        self.beta_ABS = beta_ABS
        self.beta_RA = beta_RA
        self.eps_ABS = eps_ABS  # Exploration rate for ABS model
        self.eps_RA = eps_RA    # Exploration rate for RA model
        self.c_ABS = c_ABS      # UCB uncertainty bonus factor for ABS model
        self.c_RA = c_RA        # UCB uncertainty bonus factor for RA model

        # --- Learning rates ---
        self.alphaQ_ABS = alphaQ_ABS  # Learning rate for Q-values in ABS model
        self.alphaQ_RA = alphaQ_RA    # Learning rate for Q-values in RA model
        self.alpha_Qu = alpha_Qu      # Learning rate for unchosen options
        self.alpha_exp = alpha_exp    # Rate for RA range expansion
        self.alpha_con = alpha_con    # Rate for RA range contraction
        self.gamma = gamma            # Discount factor for future rewards

        # --- Initial Q-values and Decision Policy ---
        self.Qval_init = Qval_init     # Initial Q-value for both options
        self.decision = decision       # Decision rule (e.g., 'softmax', 'UCB', 'e-greedy')

        # --- Number of Trials ---
        self.n_trial = n_trial           # Number of trials in each task

        # --- Initialize Result Storage ---
        self._initialize_results_storage(Qval_init)

    
    def _initialize_results_storage(self, Qval_init):
        """Initializes storage lists to track agent's results and performance metrics.
        
        Attributes:
            correct_ABS (list): Tracks correctness of choices in ABS model.
            Qv1_ABS, Qv2_ABS (list): Q-value histories for each option in ABS model.
            Rew_ABS, cRew_ABS (list): Actual and counterfactual rewards for ABS model choices.
            Beta_deltaQ_ABS (list): Tracks Q-value differences scaled by beta in ABS model.
        
            correct_RA (list): Tracks correctness of choices in RA model.
            Qv1_RA, Qv2_RA (list): Q-value histories for each option in RA model.
            Rmin, Rmax (list): Minimum and maximum reward estimates in RA model.
            Rew_RA, cRew_RA (list): Actual and counterfactual rewards for RA model choices.
            deltaQ_RA (list): Tracks Q-value differences in RA model.
        
            Rmax_distances, Rmin_distances (list): Distances between estimated 
                                    and true max/min rewards in RA model.
            Rmax_task_distance, Rmin_task_distance (list): Task-specific reward
                                    range distances in RA model.
            Qval_dist, Qval_task_distances (list): Distance metrics between
                                    estimated Q-values and true values.
            Qval1_dist, Qval2_dist (list): Distance metrics for individual 
                                    Q-values in relation to reward values."""
        
        # ABS Model Results
        self.correct_ABS = []
        self.Qv1_ABS = [Qval_init]
        self.Qv2_ABS = [Qval_init]
        self.Rew_ABS = []
        self.cRew_ABS = []
        self.Beta_deltaQ_ABS = []

        # RA Model Results
        self.correct_RA = []
        self.Qv1_RA = [Qval_init]
        self.Qv2_RA = [Qval_init]
        self.Rmin = [0]  # Starting with a default minimum value
        self.Rmax = [0]  # Starting with a default maximum value
        self.Rew_RA = []
        self.cRew_RA = []
        self.deltaQ_RA = []

        # Distance Metrics
        self.Rmax_distances = []
        self.Rmin_distances = []
        self.Rmax_task_distance = []
        self.Rmin_task_distance = []
        self.Qval_dist = []
        self.Qval_task_distances = []
        self.Qval1_dist = []
        self.Qval2_dist = []

    def reinit_results(self):
        """
        Resets all result-tracking attributes to their initial state for a new task.
        
        This includes clearing lists that track performance and resetting Q-values 
        and range estimates. Prepares the agent for a fresh start in the next task.
        """

        # Reset primary result storage using the same initial values as in __init__
        self._initialize_results_storage(self.Qval_init)


    def perform_task(self, task_reward):
        """
       Executes a series of trials for a given task using both ABSOLUTE and RANGE models 
       and computes performance metrics based on the distances between estimated and actual rewards.
       """
    
        # Initialize counters for ABS and RA choices
        # specific to UCB
        self.N_ABS = [0, 0] # [n_choices_opt1, n_choices_opt2]
        self.N_RAN = [0, 0] # [n_choices_opt1, n_choices_opt2]
        
        # Execute trials for both ABSOLUTE and RANGE models
        for i in range(self.n_trial):
            self.perform_trial(i, task_reward[i,:], "ABSOLUTE")
            self.perform_trial(i, task_reward[i,:], "RANGE")

        # Compute epistemic accuracy metrics
        self.Rm_avg_dist = np.mean((self.Rmax_distances, self.Rmin_distances), axis=0)
        self.Qval_task_distances = compute_task_distance(self.Qval_dist, task_reward)
        self.Qval_avg_dist = np.mean((self.Qval1_dist, self.Qval2_dist), axis=0)


    def perform_trial(self, trial_nb, trial_reward, model):
        """
        Executes a single trial for the agent under a specified decision model.
        
        Args:
            trial_nb (int): The index of the current trial within the task.
            trial_reward (ndarray): Rewards available for each option in the trial.
            model (str): Learning model to apply ('ABSOLUTE' or 'RANGE').
        
        Steps:
            - Determines the correct choice based on the highest reward.
            - Uses the specified model (ABSOLUTE or RANGE) to update Q-values and 
              distance metrics, adapting to either absolute or range-adapted values.
            - Applies a decision function (e.g., softmax, UCB) to select an action.
            - Updates result attributes based on the choice and model.
        
        Raises:
            ValueError: If an unsupported decision model is specified.
        """

        # Determine correct choice based on highest reward
        if trial_reward[1] >= trial_reward[0]:
                correct_choice = 1
        else:
                correct_choice = 0
                
        # Set initial values based on the model
        if model == "ABSOLUTE":
            self._update_absolute_model(trial_reward, correct_choice, trial_nb)
        elif model == "RANGE":
            self._update_range_model(trial_reward, correct_choice, trial_nb)
        else:
            raise ValueError(f"Unsupported model type: {model}")

    def _update_absolute_model(self, trial_reward, correct_choice, trial_nb):
        """Updates Q-values and reward metrics using the ABSOLUTE model."""
        
        Q = [self.Qv1_ABS[-1], self.Qv2_ABS[-1]]
        beta = self.beta_ABS
        eps = self.eps_ABS
        c = self.c_ABS

        # Calculate beta-adjusted Q-value difference
        # Used to find apparent beta of the range model
        # beta apparent is the abs beta value which makes beta * deltaQ_ABS closest to deltaQ_RA.
        self.Beta_deltaQ_ABS.append((Q[0] - Q[1]) * beta)

        # Compute distance between estimated Q-values and actual rewards
        self.Qval1_dist.append(dist_prop(Q[0], trial_reward[0], 
                                         np.max(trial_reward), np.min(trial_reward)))
        self.Qval2_dist.append(dist_prop(Q[1], trial_reward[1], 
                                         np.max(trial_reward), np.min(trial_reward)))
        
        # Select an action
        N = self.N_ABS 
        choice = self._apply_decision_function(Q, beta, eps, c, N, trial_nb)
        self.N_ABS[choice] += 1 # Update choice count

        # Update choice-based rewards
        reward = trial_reward[choice]
        creward = trial_reward[1 - choice]
        self.Rew_ABS.append(reward)
        self.cRew_ABS.append(creward)
        self.correct_ABS.append(int(choice == correct_choice))

        # Update Q-values based on prediction error
        pred_error = reward - Q[choice]
        Q[choice] = Q[choice] + self.alphaQ_ABS * pred_error

        self.Qv1_ABS.append(Q[0])
        self.Qv2_ABS.append(Q[1])

    
    def _update_range_model(self, trial_reward, correct_choice, trial_nb):
        """Updates Q-values and reward metrics using the RANGE model."""
        
        Q = [self.Qv1_RA[-1], self.Qv2_RA[-1]]
        beta = self.beta_RA
        eps = self.eps_RA
        c = self.c_RA

        # Track Q-value differences for RA model
        self.deltaQ_RA.append((Q[0] - Q[1]))

        # Calculate distances to estimated range boundaries
        self.Rmax_distances.append(dist_prop(self.Rmax[-1], np.max(trial_reward),
                                             np.max(trial_reward), np.min(trial_reward)))
        self.Rmin_distances.append(dist_prop(self.Rmin[-1], np.min(trial_reward),
                                             np.max(trial_reward), np.min(trial_reward)))

        # Select an action
        N = self.N_RAN
        choice = self._apply_decision_function(Q, beta, eps, c, N, trial_nb)
        self.N_RAN[choice] += 1 # Update choice count

        # Update rewards and correctness
        reward = trial_reward[choice]
        creward = trial_reward[1 - choice]
        self.Rew_RA.append(reward)
        self.cRew_RA.append(creward)
        self.correct_RA.append(int(choice == correct_choice))

        # Update Rmax and Rmin based on reward
        delta_R = reward - self.Rmax[-1]
        delta_r = reward - self.Rmin[-1]
        self.Rmax.append(self.Rmax[-1] + self.alpha_exp * delta_R * (delta_R > 0) +
                         self.alpha_con * delta_R * (delta_R < 0))
        self.Rmin.append(self.Rmin[-1] + self.alpha_exp * delta_r * (delta_r < 0) +
                         self.alpha_con * delta_r * (delta_r > 0))

        # Calculate reward relative to estimated range
        if self.Rmin[-1] != self.Rmax[-1]:
            rel_reward = (reward - self.Rmin[-1]) / (self.Rmax[-1] - self.Rmin[-1])
        else:
            rel_reward = reward
            
        # Update Q-values with prediction error based on relative reward
        pred_error = rel_reward - Q[choice] if self.decision != 'policy' else reward - Q[choice]
        Q[choice] += self.alphaQ_RA * pred_error
        self.Qv1_RA.append(Q[0])
        self.Qv2_RA.append(Q[1])

    def _apply_decision_function(self, Q, beta, eps, c, N, trial_nb):
        """Applies the selected decision function to choose an action."""
    
        if self.decision == 'softmax':
            return decision_function('softmax', Q[0], Q[1], [beta], trial_nb)
        elif self.decision == 'e-greedy':
            return decision_function('e-greedy', Q[0], Q[1], [eps], trial_nb)
        elif self.decision == 'UCB':
            return decision_function('UCB', Q[0], Q[1], 
                                     [c, N[0], N[1], trial_nb], trial_nb)
        elif self.decision == 'policy':
            if Q[0] + Q[1] != 0:
                return decision_function('softmax', Q[0] / (Q[0] + Q[1]), 
                                         Q[1] / (Q[0] + Q[1]), [beta], trial_nb)
            else:
                return decision_function('softmax', Q[0], Q[1], [beta], trial_nb)
        else:
            raise ValueError(f"Invalid decision type: {self.decision}")
