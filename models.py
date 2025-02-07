#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the `Model` class, which simulates agent-based decision-making 
experiments over multiple tasks and trials. Parameters can  be set for individual agents.
The model class handles the configuration of parameter spaces, execution of simulations,
and storage of results for further analysis.

Classes:
    - Model: A class for setting up and running simulations of multiple agents performing 
             decision-making tasks with adjustable learning parameters and decision policies.
             
Key Methods:
    - set_param_space: Configures the parameter space based on input factors, with options 
                       for sequential, exhaustive, or random modes.
    - go: Runs the simulation loop for all agents and tasks, storing and optionally aggregating 
          results based on the selected mode.
    - init_storage: Initializes storage arrays for capturing simulation results across trials, 
                    tasks, and agents.
    - store_results: Saves the results of each agent's task performance.

Dependencies:
    - numpy
    - progressbar
    - Agent (from Agents.py)
    - general_functions_B

Created on Thu Dec  9 23:11:17 2021

@author: jeremyperez, Maëva l'Hôtellier
"""

import numpy as np
import progressbar

from agents import Agent
from general_functions import create_reward


class Model:
    def __init__(self, n_subjects, n_trial, random_learning_rates=True, 
                 agent_params=None, beta_ABS=1, beta_RA=1, Qval_init=0, 
                 eps_ABS=0.05, eps_RA=0.05, c_ABS=1, c_RA=1, 
                 decision='softmax', talk=True):
        """"
        A model class for simulating multiple agents interacting in a 
        series of trials with customizable learning and decision-making parameters.

        Attributes:
           - n_subjects (int): Number of agents in the simulation.
           - n_trial (int): Number of trials in the simulation.
           - random_learning_rates (bool): Flag to randomly initialize learning rates.
                                          
             If random_learning_rates is False, you should provide agent_params:
           - agent_params (list, optional): List of dictionnaries containing the
                                        parameters values specific to each agent 
                                        (see Agent class for details).
                                        Example:
                                                agent_params = {
                                                    'alphaQ_ABS': 0.3, 
                                                    'alphaQ_RA': 0.3,
                                                    'alpha_Qu': None,
                                                    'alpha_exp': 0.75, 
                                                    'alpha_con': 0.2, 
                                                    'gamma': None,
                                                    'Qval_init': 0,
                                                    'decision': 'softmax', 
                                                    'beta_ABS': 1, 'beta_RA': 1,
                                                    'eps_ABS': None, 'eps_RA': None, 
                                                    'c_ABS': None, 'c_RA': None
                                                    }
                                            
                 If random_learning_ratesis True, the following arguments will be used:
            -    beta_ABS, beta_RA (float): Temperature for Softmax policy.
            -    Qval_init (float): Initial Q-value for agents' choices.
            -    eps_ABS (float): e-greedy epsilon value for ABS agents.
            -    eps_RA (float):  e-greedy epsilon value for RA agents.
            -    c_ABS (float): 'c' value for the UCB policy specific to ABS agents.
            -    c_RA (float): 'c' value for the UCB policy specific to RA agents.
            -    decision (str): Decision rule, can be 'softmax', 'UCB' or 'e-greedy'.

            - talk (bool): Flag for enabling print statements.

        
        Raises:
            ValueError: If any parameter is invalid or outside the expected range.
        """

        # Validate input parameters
        if not isinstance(n_subjects, int) or n_subjects <= 0:
            raise ValueError("n_subjects must be a positive integer")
        if not isinstance(n_trial, int) or n_trial <= 0:
            raise ValueError("n_trial must be a positive integer")
        if agent_params is not None and not isinstance(agent_params[0], dict):
            raise ValueError(
                "agent_params must be a list of dictionnaries if provided. "
                "Expected order of parameters: [beta_ABS, beta_RA, alphaQ_ABS, alphaQ_RA, "
                "alpha_Qu, alpha_exp, alpha_con, gamma]")

        self.n_subjects = n_subjects
        self.n_trial = n_trial
        self.agent_params = []

        # Initialize learning rates based on random or provided values
        if random_learning_rates:
            self._initialize_random_learning_rates(beta_ABS, beta_RA, Qval_init, 
                                                   eps_ABS, eps_RA, 
                                                   c_ABS, c_RA, decision)

        else:

            self.agent_params = agent_params

        self.talk = talk
   
    def _initialize_random_learning_rates(self, beta_ABS, beta_RA, Qval_init, 
                                          eps_ABS, eps_RA, c_ABS, c_RA, decision):
        """
        Initializes random learning rates for each agent.
        """
        for _ in range(self.n_subjects):
            alphaQ_ABS = np.random.random() # Learning rate for ABS Q-values
            alphaQ_RA = np.random.random() # Learning rate for  RA Q-values
            alpha_exp = np.random.random() # Learning rate for RA range expansion
            alpha_con = max(0, np.random.random() * alpha_exp)  # Learning rate for RA range contraction

            # Append initialized parameters for each agent
            self.agent_params.append({
                'beta_ABS': beta_ABS, 
                'beta_RA': beta_RA, 
                'alphaQ_ABS': alphaQ_ABS, 
                'alphaQ_RA': alphaQ_RA, 
                'alpha_Qu': None,
                'alpha_exp': alpha_exp, 
                'alpha_con': alpha_con, 
                'gamma': None, 
                'Qval_init': Qval_init, 
                'eps_ABS': eps_ABS,
                'eps_RA': eps_RA,
                'c_ABS':c_ABS, 
                'c_RA': c_RA,
                'decision': decision})


    def set_param_space(self, rescale, noise, discrim, dsigma,
                        mode="exhaustive", n_random=None, seed=None):
        """
        Configures the parameter space for agent tasks based on input factors.
        
        Args:
            rescale (list): Scaling factors applied to rewards.
            noise (list): Noise level injected in the rewards.
            discrim (list): Discriminability level between rewards.
            dsigma (list): Variance values for random walks.
            mode (str, optional): Mode for setting parameter space; 
                                 - "sequential": Parameter configurations are
                                  generated by iterating over each parameter 
                                  independently in a pre-defined order .
                                 - "exhaustive": Generates every possible 
                                  combination of all parameter values.
                                 - "random": Samples configurations randomly 
                                    from the parameter space with `n_random` 
                                    configurations using optional `seed`.
            n_random (int, optional): Number of random configurations in "random" mode.
            seed (int, optional): Seed for reproducibility in "random" mode.
    
        Attributes Set:
            param_space (list): Configured parameter space.
            n_task (int): Total number of tasks in the parameter space.
        """

        if not all(isinstance(lst, (list, np.ndarray)) for lst in [rescale, noise, discrim, dsigma]):
            raise ValueError("rescale, noise, discrim, and dsigma must all be lists or numpy arrays.")

        if not all(isinstance(lst, (list, np.ndarray)) and len(lst) > 0 for lst in [rescale, noise, discrim, dsigma]):
            raise ValueError("rescale, noise, discrim, and dsigma must not be empty and must be lists or numpy arrays.")


        self.param_space = []
        self.mode = mode
        self.rescale = rescale
        self.discrim = discrim

        if mode == "sequential":
            for r in rescale:
                self.param_space.append([r, 0, 0.4, 0])
            self.ind_rescale = len(rescale)

            for n in noise:
                self.param_space.append([1, n, 0.4, 0])
            self.ind_noise = len(noise) + self.ind_rescale
            
            for d in discrim:
                self.param_space.append([1, 0, d, 0])
            self.ind_discrim = len(discrim) + self.ind_noise
            self.n_contexts = self.ind_discrim + len(dsigma)

            for dsig in dsigma:
                for i in range(10):
                    self.param_space.append([1, 0, 0.4, dsig])

            self.n_task = len(self.param_space)

        
        elif mode == "exhaustive":
            for r in rescale:
                for n in noise:
                    for d in discrim:
                        for dsig in dsigma:
                            self.param_space.append([r, n, d, dsig])

            self.n_task = len(self.param_space)
                            

        elif mode == "random":
            np.random.seed(seed)
            for i in range(n_random):
                self.param_space.append([np.random.choice(rescale), 
                                         np.random.choice(noise), 
                                         np.random.choice(discrim),
                                         np.random.choice(dsigma)])
            self.n_task = n_random
        
        self.param_space = np.array(self.param_space)


       # Talk mode output
        if self.talk:
            print("Space of parameters:")
            print(f"Rescale range: [{rescale[0]}, {rescale[-1]}]")
            print(f"Noise range: [{noise[0]}, {noise[-1]}]")
            print(f"Discriminability range: [{discrim[0]}, {discrim[-1]}]")
            print(f"Variance (dsigma) range: [{dsigma[0]}, {dsigma[-1]}]")
            print(f"Mode: {mode}")
            print(f"Each of the {self.n_subjects} agents will perform " + 
                  f"{self.n_task} tasks of {self.n_trial} trials")

        self.init_storage()   


    def init_storage(self):
        """
        Initializes storage arrays for simulation results across trials, tasks, and subjects.
        
        Attributes:
            
            Specific to ABS agents:
                correct_ABS (ndarray): Tracks correctness for each trial and task.
                Qv1_ABS, Qv2_ABS (ndarray): Q-values across trials and tasks.
                Rew_ABS, cRew_ABS (ndarray): Obtained and counterfactual rewards for ABS agents.
                Qval_avg_dist (ndarray): Distance between true values and ABS agent Q-values.
                Agents_Qval_dist (ndarray): Average Q-value distance across trials for ABS agents.
                Qval_task_distances (ndarray): Q-value distances for tasks (if applicable).

            Specific to RA agents:
                correct_RA (ndarray): Tracks correctness of RA agents across trials and tasks.
                Qv1_RA, Qv2_RA (ndarray): Q-values for RA agents across trials and tasks.
                Rmin, Rmax (ndarray): Estimated reward range (min, max) for RA agents.
                Rew_RA, cRew_RA (ndarray): Obtained and counterfactual rewards for RA agents.
                Rmax_distances, Rmin_distances (ndarray): Distances between estimated and true range.
                Rm_avg_dists (ndarray): Average distances between estimated min and max.
                Agents_Rm_dist (ndarray): Average reward distance across trials for RA agents.

            Beta_deltaQ_ABS, deltaQ_RA (ndarray): 
        """
        # Define array shapes
        shape_3d = (self.n_trial, self.n_task, self.n_subjects)
        shape_2d = (self.n_task, self.n_subjects)

       # Storage specific to ABS agents
        self.correct_ABS = np.zeros(shape_3d)
        self.Qv1_ABS = np.zeros(shape_3d)
        self.Qv2_ABS = np.zeros(shape_3d)
        self.Rew_ABS = np.zeros(shape_3d)
        self.cRew_ABS = np.zeros(shape_3d)
        self.Qval_avg_dist = np.zeros(shape_3d)
        self.Agents_Qval_dist = np.zeros(shape_2d)
        self.Qval_task_distances = np.zeros(shape_2d)
        self.Beta_deltaQ_ABS = np.zeros(shape_3d)
        
        # Storage specific to RA agents
        self.correct_RA = np.zeros(shape_3d)
        self.Qv1_RA = np.zeros(shape_3d)
        self.Qv2_RA = np.zeros(shape_3d)
        self.Rmin = np.zeros(shape_3d)
        self.Rmax = np.zeros(shape_3d)
        self.Rew_RA = np.zeros(shape_3d)
        self.cRew_RA = np.zeros(shape_3d)
        self.Rmax_distances = np.zeros(shape_3d)
        self.Rmin_distances = np.zeros(shape_3d)
        self.Rm_avg_dists = np.zeros(shape_3d)
        self.Agents_Rm_dist = np.zeros(shape_2d)
        self.deltaQ_RA = np.zeros(shape_3d)
    
        
    def store_results(self, agent, agent_nb, task_nb):
        """
        Stores simulation results for a specific agent and task.
    
        Args:
            agent (Agent): The agent object containing simulation results.
            agent_nb (int): Index of the agent in the population.
            task_nb (int): Index of the task in the sequence (we call it an "environment" in the paper).
        
        Updates:
            - correct_ABS, Qv1_ABS, Qv2_ABS, Rew_ABS, cRew_ABS: Stores ABS agent results.
            - correct_RA, Qv1_RA, Qv2_RA, Rmin, Rmax, Rew_RA, cRew_RA: Stores RA agent results.
            - Rmax_distances, Rmin_distances, Rm_avg_dists, Qval_avg_dist: Distance metrics.
            - Agents_Rm_dist, Agents_Qval_dist: Average distances across trials.
            - Beta_deltaQ_ABS, deltaQ_RA: Q-value differences across strategies.
        """
        # Stores results specific to ABS agents
        self.correct_ABS[:, task_nb, agent_nb] = agent.correct_ABS
        self.Qv1_ABS[:, task_nb, agent_nb] = agent.Qv1_ABS[:-1]
        self.Qv2_ABS[:, task_nb, agent_nb] = agent.Qv2_ABS[:-1]
        self.Rew_ABS[:, task_nb, agent_nb] = agent.Rew_ABS
        self.cRew_ABS[:, task_nb, agent_nb] = agent.cRew_ABS
        self.Agents_Qval_dist[task_nb, agent_nb] = np.mean(agent.Qval_avg_dist)        
        self.Beta_deltaQ_ABS[:, task_nb, agent_nb] = agent.Beta_deltaQ_ABS
        self.Qval_task_distances[task_nb, agent_nb] = agent.Qval_task_distances
        
        # Stores results specific to RA agents
        self.correct_RA[:, task_nb, agent_nb] = agent.correct_RA
        self.Qv1_RA[:, task_nb, agent_nb] = agent.Qv1_RA[:-1]
        self.Qv2_RA[:, task_nb, agent_nb] = agent.Qv2_RA[:-1]
        self.Rmin[:, task_nb, agent_nb] = agent.Rmin[:-1]
        self.Rmax[:, task_nb, agent_nb] = agent.Rmax[:-1]
        self.Rew_RA[:, task_nb, agent_nb] = agent.Rew_RA
        self.cRew_RA[:, task_nb, agent_nb] = agent.cRew_RA
        self.Rmax_distances[:, task_nb, agent_nb] = agent.Rmax_distances
        self.Rmin_distances[:, task_nb, agent_nb] = agent.Rmin_distances
        self.Rm_avg_dists[:, task_nb, agent_nb] = agent.Rm_avg_dist
        self.Qval_avg_dist[:, task_nb, agent_nb] = agent.Qval_avg_dist
        self.Agents_Rm_dist[task_nb, agent_nb] = np.mean(agent.Rm_avg_dist)
        self.deltaQ_RA[:, task_nb, agent_nb] = agent.deltaQ_RA

            
    def go(self):
        """
        Runs the main simulation loop across all tasks and agents,
        storing and aggregating results.

        Steps:
            - Initializes rewards for all tasks.
            - Iterates over each agent, allowing them to perform all tasks based 
              on initialized parameters.
            - Aggregates results by `dsigma` values when in `sequential` mode.
        
        Attributes:
            all_task_rewards (ndarray): Stores reward values for each task.
        
        """

        if self.talk:
            print("\n Progress :")
            bar = progressbar.ProgressBar(maxval=self.n_subjects, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        
        # Initialize rewards and progress bar
        self.all_task_rewards = np.zeros((self.n_task, self.n_trial, 2))
        for task in range(self.n_task):
            [rescale, noise, discrim, dsigma ] = self.param_space[task]
            self.all_task_rewards[task,:, :] = create_reward(rescale, noise, 
                                                             discrim, dsigma, 
                                                             self.n_trial)
        # Loop through each agent and task
        for agent_nb in range(self.n_subjects):
            if self.talk:
                bar.update(agent_nb)

            agent = Agent(**self.agent_params[agent_nb], 
                          n_trial=self.n_trial)           
            # Agent performs tasks and results are stored
            for task in range(self.n_task):
                # Retrieve task parameters
                [rescale, noise, discrim, dsigma ] = self.param_space[task]
                # Perform task
                agent.perform_task(self.all_task_rewards[task, :, :])
                # Store results in the model object
                self.store_results(agent, agent_nb, task)
                # Reset agent results for the next task
                agent.reinit_results()
                
        if self.mode == "sequential":
            
            # Number of unique dsigma values, assuming 10 identical tasks per dsigma
            n_dsig = int(len(self.param_space[self.ind_discrim:, 3]) / 10)
            print(n_dsig)
            aggr_dsigma = np.zeros(n_dsig)
            aggr_correct_ABS = np.zeros((self.n_trial, n_dsig, self.n_subjects))
            aggr_correct_RA = np.zeros((self.n_trial, n_dsig, self.n_subjects))
            
            # Aggregate correctness by averaging over identical dsigma tasks
            for i in range(n_dsig):
                aggr_dsigma[i] = self.param_space[self.ind_discrim + i * 10 ,3]
                #print(aggr_dsigma[i])
                aggr_correct_ABS[:,i,:] = np.mean(
                    self.correct_ABS[:,self.ind_discrim + i * 10: 
                                 self.ind_discrim + (i+1) * 10 ,:],axis = (1))
                aggr_correct_RA[:,i,:] = np.mean(
                    self.correct_RA[:,self.ind_discrim + i * 10: 
                                  self.ind_discrim + (i+1) * 10 ,:],axis = (1))

            # Update parameter space and correctness arrays with aggregated values
            self.param_space = self.param_space[:self.n_contexts,:]
            self.correct_ABS = self.correct_ABS[:,:self.n_contexts,:]
            self.correct_RA = self.correct_RA[:,:self.n_contexts,:]
            self.param_space[:,3] = np.concatenate((self.param_space[
                :self.ind_discrim, 3], aggr_dsigma))
            self.correct_ABS[:, self.ind_discrim:,:] = aggr_correct_ABS
            self.correct_RA[:, self.ind_discrim:,:] = aggr_correct_RA
        
        if self.talk:
            bar.finish()