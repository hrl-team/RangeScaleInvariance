"""4x4 gridworld with two terminal rewards (Fig. 7).
Called from 'notebooks/Figure7.ipynb'.

Layout 
------
(*, 0, 0, s)
(0, 0, 0, 0)
(0, 0, 0, 0)
(0, 0, 0, b)

* = starting state
s = small reward (three steps away , "proximal")
b = big reward ( six steps away, "distal")

All other transitions pay nothing. Actions that would leave the grid leave the
agent where it is.

Because the proximal terminal is closer, a random policy does not end equally
often in the two: it reaches (0, 3) on 57.5% of episodes and (3, 3) on 42.5%.
Chance level for "reaching the global optimum" is 0.425 when the larger reward
sits at the distal cell, and 0.575 when it sits at the proximal one.

Learning
--------
Both models are tabular Q-learning. They differ only in the target:

    ABS   terminal     target = r
          non-terminal target = 0 + gamma * max_a Q(s')

    RA    terminal     target = s(r)
          non-terminal target =     gamma * max_a Q(s')

with s(r) = (r - Rmin) / (Rmax - Rmin). The range is updated only when a
non-zero reward is received, i.e. on arrival at a terminal state.

Implementation
--------------
All agents are stepped together, each with its own Q-table, so one call
simulates a whole population. Episodes end at different times, so a boolean
mask tracks which agents are still running.
"""

from __future__ import annotations

import numpy as np

__all__ = ["run_gridworld", "CHANCE_DISTAL", "CHANCE_PROXIMAL", "terminal_rewards"]

GRID = 4
N_ACTIONS = 4                       # 0: up, 1: down, 2: left, 3: right
PROXIMAL, DISTAL = (0, 3), (3, 3)

CHANCE_DISTAL = 0.425               # random policy reaches (3, 3) this often
CHANCE_PROXIMAL = 0.575


def terminal_rewards(magnitude: float, discriminability: float):
    """Payoffs of the two terminal states, defined as in the bandit task.

    Returns `(r_proximal, r_distal)`: the short path pays less, the long path
    more, and `discriminability` sets how far apart they are.
    """
    return ((0.5 - discriminability / 2) * magnitude,
            (0.5 + discriminability / 2) * magnitude)


# Transition table, precomputed once: _NEXT[row, col, action] -> (row, col).
_NEXT = np.empty((GRID, GRID, N_ACTIONS, 2), dtype=np.int64)
for _r in range(GRID):
    for _c in range(GRID):
        _NEXT[_r, _c, 0] = (max(0, _r - 1), _c) # up
        _NEXT[_r, _c, 1] = (min(GRID - 1, _r + 1), _c) # down
        _NEXT[_r, _c, 2] = (_r, max(0, _c - 1)) # left
        _NEXT[_r, _c, 3] = (_r, min(GRID - 1, _c + 1)) # right

# _NEXT gives the next cell for each action, but some actions hit walls. The following
#    shows the example behavior of each action from each cell:
#    from (0, 0)                      from (1, 1)
#    up    -> (0, 0)   wall             up   -> (0, 1)
#    down  -> (1, 0)                    down -> (2, 1)
#    left  -> (0, 0)   wall             left -> (1, 0)
#    right -> (0, 1)                   right -> (1, 2)

def run_gridworld(r_proximal, r_distal, model, n_agents=100, n_episodes=300,
                  burn_in=150, beta=1.0, gamma=0.99, max_steps=500, seed=0):
    """Simulate `n_agents` independent learners on one gridworld.

    Each agent gets its own learning rates, drawn as in the bandit simulations:
    alpha_Q uniform, alpha_exp uniform, and alpha_con = alpha_exp * uniform so
    that contraction is always slower than expansion.

    Returns
    -------
    (n_agents,) array: the proportion of post-burn-in episodes that ended at the
    larger of the two terminal states.
    """
    if model not in ("ABS", "RA"):
        raise ValueError(f"model must be 'ABS' or 'RA', got {model!r}")

    rng = np.random.default_rng(seed)
    goal = DISTAL if r_distal >= r_proximal else PROXIMAL

    # Reward lookup by cell, zero everywhere except the two terminals.
    reward_map = np.zeros((GRID, GRID))
    reward_map[PROXIMAL] = r_proximal
    reward_map[DISTAL] = r_distal
    is_terminal = reward_map != 0.0

    alpha_q = rng.random(n_agents)
    alpha_exp = rng.random(n_agents)
    alpha_con = alpha_exp * rng.random(n_agents)

    q = np.zeros((n_agents, GRID, GRID, N_ACTIONS))
    r_min = np.zeros(n_agents)
    r_max = np.zeros(n_agents)

    agents = np.arange(n_agents)
    hits = np.zeros(n_agents)
    counted = 0

    for episode in range(n_episodes):
        row = np.zeros(n_agents, dtype=np.int64) # tracks the current row of each agent
        col = np.zeros(n_agents, dtype=np.int64) # tracks the current column of each agent
        active = np.ones(n_agents, dtype=bool) # tracks which agents are still running
        ended_at_goal = np.zeros(n_agents, dtype=bool) # tracks which agents ended at the larger terminal

        for _ in range(max_steps):
            if not active.any():
                break
            idx = agents[active]
            r_, c_ = row[idx], col[idx]

            # --- softmax over the four actions, stabilised ------------------ #
            qs = q[idx, r_, c_] * beta # scale by temperature
            qs -= qs.max(axis=1, keepdims=True) # subtract max for numerical stability
            probs = np.exp(qs) # exponentiate
            probs /= probs.sum(axis=1, keepdims=True) # normalise to get probabilities
            action = (probs.cumsum(axis=1) >= rng.random((idx.size, 1))).argmax(axis=1) # sample an action for each agent

            nxt = _NEXT[r_, c_, action] # get the next cell for each agent based on their action
            nr, nc = nxt[:, 0], nxt[:, 1] # update the row and column for each agent based on the next cell
            reward = reward_map[nr, nc] # get the reward for each agent based on their new position (either 0, r_proximal, or r_distal)
            done = is_terminal[nr, nc] # check if each agent has reached a terminal state

            # --- target ----------------------------------------------------- #
            future = gamma * q[idx, nr, nc].max(axis=1) # compute the future value for each agent based on their new position
            if model == "ABS":
                # The absolute model uses the raw reward for terminal states
                # and the discounted future value for non-terminal states.
                target = np.where(done, reward, reward + future)
            else:
                # The range moves only on terminal (non-zero) rewards.
                got = reward != 0.0
                if got.any():
                    g = idx[got] # indices of agents that got a non-zero reward
                    d_max = reward[got] - r_max[g] # update the max range for agents that got a non-zero reward
                    d_min = reward[got] - r_min[g] # update the min range for agents that got a non-zero reward
                    r_max[g] += np.where(d_max > 0, alpha_exp[g], alpha_con[g]) * d_max
                    r_min[g] += np.where(d_min < 0, alpha_exp[g], alpha_con[g]) * d_min
                span = r_max[idx] - r_min[idx]
                rel = np.where(span != 0,
                               (reward - r_min[idx]) / np.where(span != 0, span, 1.0),
                               reward)
                # compute the target for each agent based on whether they reached a terminal state or not
                target = np.where(done, rel, future) 

            # --- update Q-table ------------------------------------------------ #
            q[idx, r_, c_, action] += alpha_q[idx] * (target - q[idx, r_, c_, action])

            row[idx], col[idx] = nr, nc # update the row and column for each agent based on their new position
            finished = idx[done] # indices of agents that have reached a terminal state
            # check if each agent that reached a terminal state reached the larger terminal state
            ended_at_goal[finished] = (nr[done] == goal[0]) & (nc[done] == goal[1])
            active[finished] = False # mark agents that have reached a terminal state as inactive

        # count only post-burn-in episodes
        if episode >= burn_in: 
            hits += ended_at_goal
            counted += 1

    return hits / max(counted, 1)
