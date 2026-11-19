"""The seasonal gridworld of Fig. 7D-F.

Called from 'notebooks/Figure7.ipynb'.
This is a modified version of the gridworld environment where
the location of the best rewards change every `season_length` episodes,
together with the overall reward magnitude.
Q-tables and  Rmin and Rmax carry over across the boundary, nothing is reset.

That is the point: what an agent has learned in one
season is partly wrong in the next, and the question is how fast it recovers.
"""

from __future__ import annotations

import numpy as np

from .gridworld import (CHANCE_DISTAL, CHANCE_PROXIMAL, DISTAL, GRID,
                        N_ACTIONS, PROXIMAL, _NEXT)

__all__ = ["make_season_rewards", "run_seasons", "CHANCE_DISTAL",
           "CHANCE_PROXIMAL", "DISCRIMINABILITY"]

DISCRIMINABILITY = 0.25


def make_season_rewards(magnitudes, n_agents, shuffle=True, seed=0):
    """Reward values and optimum position for every agent and season.

    Each agent gets its own order of the magnitudes when `shuffle` is set, and
    the larger reward alternates between the two cells from one season to the
    next.

    Returns
    -------
    rewards : (n_agents, n_seasons, 2) array
        Reward at the proximal cell and at the distal cell.
    big_at_distal : (n_agents, n_seasons) bool array
    magnitude : (n_agents, n_seasons) array
        The magnitude each agent saw in each season, in the order it saw it.
    """
    rng = np.random.default_rng(seed)
    magnitudes = np.asarray(magnitudes, dtype=float)
    n_seasons = magnitudes.size

    order = np.empty((n_agents, n_seasons)) # the order of magnitudes for each agent
    for a in range(n_agents):
        order[a] = rng.permutation(magnitudes) if shuffle else magnitudes

    season = np.arange(n_seasons)[None, :]
    parity = (np.arange(n_agents) % 2 == 0)[:, None] # even or odd
    big_at_distal = np.where(parity, season % 2 == 1, season % 2 == 0) 
    # True if the larger reward is at the distal cell, False if at the proximal cell

    r_big = order * (0.5 + DISCRIMINABILITY / 2) # the larger reward value for each agent and season
    r_small = order * (0.5 - DISCRIMINABILITY / 2) # the smaller reward value for each agent and season
    rewards = np.empty((n_agents, n_seasons, 2))
    rewards[:, :, 0] = np.where(big_at_distal, r_small, r_big)   # proximal
    rewards[:, :, 1] = np.where(big_at_distal, r_big, r_small)   # distal
    return rewards, big_at_distal, order


def run_seasons(rewards, season_length, beta=1.0, gamma=0.99, max_steps=500,
                seed=0):
    """Run both models through a sequence of seasons.

    Parameters
    ----------
    rewards : (n_agents, n_seasons, 2)
        Output of :func:`make_season_rewards`.
    season_length : int
        Episodes per season.

    Returns
    -------
    dict with ``ABS`` and ``RA``, each an (n_agents, n_episodes) boolean array
    recording whether the episode ended at the larger of the two terminals.
    """
    rng = np.random.default_rng(seed)
    n_agents, n_seasons, _ = rewards.shape
    n_episodes = n_seasons * season_length

    # Learning rates as in GridWorldAgent(random_learning_rates=True).
    alpha_q = {"ABS": rng.random(n_agents), "RA": rng.random(n_agents)}
    alpha_exp = rng.uniform(0.1, 0.99, n_agents)
    alpha_con = rng.random(n_agents) * alpha_exp

    ## Q-tables and Rmin/Rmax for each model, agent, and cell.
    q = {m: np.zeros((n_agents, GRID, GRID, N_ACTIONS)) for m in ("ABS", "RA")}
    r_min = np.zeros(n_agents)
    r_max = np.zeros(n_agents)

    agents = np.arange(n_agents) # indices of agents
    out = {m: np.zeros((n_agents, n_episodes), dtype=bool) for m in ("ABS", "RA")}

    ## --- run the episodes ----------------------------------------------------- #
    for episode in range(n_episodes):
        season = episode // season_length # which season we are in
        r_prox = rewards[:, season, 0] # proximal reward for each agent
        r_dist = rewards[:, season, 1] # distal reward for each agent
        goal_is_distal = r_dist > r_prox

        for model in ("ABS", "RA"):
            row = np.zeros(n_agents, dtype=np.int64)
            col = np.zeros(n_agents, dtype=np.int64)
            active = np.ones(n_agents, dtype=bool)

            for _ in range(max_steps):
                if not active.any():
                    break
                idx = agents[active]
                r_, c_ = row[idx], col[idx]

                qs = q[model][idx, r_, c_] * beta
                qs -= qs.max(axis=1, keepdims=True) # subtract max for numerical stability
                probs = np.exp(qs)
                probs /= probs.sum(axis=1, keepdims=True)
                action = (probs.cumsum(axis=1)
                          >= rng.random((idx.size, 1))).argmax(axis=1)

                nxt = _NEXT[r_, c_, action] # get the next cell for each agent based on their action
                nr, nc = nxt[:, 0], nxt[:, 1]

                # check if each agent has reached the proximal terminal state
                at_prox = (nr == PROXIMAL[0]) & (nc == PROXIMAL[1])
                at_dist = (nr == DISTAL[0]) & (nc == DISTAL[1])
                done = at_prox | at_dist
                reward = np.where(at_prox, r_prox[idx],
                                  np.where(at_dist, r_dist[idx], 0.0))

                # compute the future value for each agent based on their new position
                future = gamma * q[model][idx, nr, nc].max(axis=1)
                if model == "ABS":
                    # see the comment in the gridworld.py file for the ABS model
                    target = np.where(done, reward, reward + future)
                else:
                    # see the comment in the gridworld.py file for the RA model
                    # The range moves only on terminal (non-zero) rewards, and
                    # survives the season boundary.
                    got = reward != 0.0
                    if got.any():
                        g = idx[got]
                        d_max = reward[got] - r_max[g]
                        d_min = reward[got] - r_min[g]
                        r_max[g] += np.where(d_max > 0, alpha_exp[g],
                                             alpha_con[g]) * d_max
                        r_min[g] += np.where(d_min < 0, alpha_exp[g],
                                             alpha_con[g]) * d_min
                    span = r_max[idx] - r_min[idx]
                    rel = np.where(span != 0,
                                   (reward - r_min[idx])
                                   / np.where(span != 0, span, 1.0),
                                   reward)
                    target = np.where(done, rel, future)

                # --- update Q-table ------------------------------------------------ #
                q[model][idx, r_, c_, action] += (
                    alpha_q[model][idx] * (target - q[model][idx, r_, c_, action]))

                row[idx], col[idx] = nr, nc
                finished = idx[done]
                reached_distal = at_dist[done]
                out[model][finished, episode] = np.where(
                    goal_is_distal[finished], reached_distal, ~reached_distal)
                active[finished] = False

    return out


def rolling_mean(series, window=15):
    """Trailing rolling mean along the last axis (n_episodes)"""
    series = np.asarray(series, dtype=float)
    kernel = np.ones(window) / window
    # Pad the series with the first value so that the rolling mean
    # is defined for the first `window-1` episodes.
    padded = np.concatenate(
        [np.repeat(series[..., :1], window - 1, axis=-1), series], axis=-1)
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), -1, padded)
