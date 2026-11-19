"""The non-stationary bandits of Fig. 4.

Called in 'notebooks/Figure4.ipynb'.

Four environments, all of 80 trials.

    expanding    the better option pays 0.6, 0.7, 0.8 then 0.9, in blocks of 20
    shrinking    the same blocks in reverse, 0.9 down to 0.6
    slow drift   a Gaussian random walk on both options, sigma = 0.05
    fast drift   the same walk with sigma = 0.1
"""

from __future__ import annotations

import numpy as np

__all__ = ["SCENARIOS", "N_TRIALS", "make_dynamic_rewards"]

N_TRIALS = 80
WALK_SEED = 10

SCENARIOS = ("expanding", "shrinking", "slow drift", "fast drift")
ALL_SCENARIOS = ("stationary",) + SCENARIOS

# Panel letters of Fig. 4.
PANEL_LETTERS = {"expanding": "C", "shrinking": "D",
                 "slow drift": "E", "fast drift": "F"}

# Titles used in the published figure.
PANEL_TITLES = {"expanding": "Expanding Bandit", "shrinking": "Shrinking Bandit",
                "slow drift": "Slow drifting Bandit",
                "fast drift": "Fast drifting Bandit"}


def _blocks(values, n_trials=N_TRIALS):
    """Reward of the better option, held constant over equal blocks."""
    per_block = n_trials // len(values)
    better = np.repeat(values, per_block).astype(float)
    return np.column_stack((1.0 - better, better))


def _random_walk(discriminability, dsigma, n_trials=N_TRIALS, seed=WALK_SEED):
    """Both options take an independent Gaussian step each trial, clipped to [0, 1].
    """
    rewards = np.empty((n_trials, 2))
    rewards[:, 0] = 0.5 - discriminability / 2
    rewards[:, 1] = 0.5 + discriminability / 2

    rng = np.random.RandomState(seed)
    for t in range(1, n_trials):
        step = rng.normal(0, dsigma, size=(2, 1))
        rewards[t, 0] = np.clip(rewards[t - 1, 0] + step[0], 0, 1)
        rewards[t, 1] = np.clip(rewards[t - 1, 1] + step[1], 0, 1)
    return rewards


def make_dynamic_rewards(scenario: str, n_trials: int = N_TRIALS) -> np.ndarray:
    """Reward sequence of one scenario, shape (n_trials, 2).
    Column 0 is the worse option, column 1 the better one.
    """
    if scenario == "stationary":
        return _blocks([0.6], n_trials)
    if scenario == "expanding":
        return _blocks([0.6, 0.7, 0.8, 0.9], n_trials)
    if scenario == "shrinking":
        return _blocks([0.9, 0.8, 0.7, 0.6], n_trials)
    if scenario == "slow drift":
        return _random_walk(0.5, 0.05, n_trials)
    if scenario == "fast drift":
        return _random_walk(0.5, 0.1, n_trials)
    raise ValueError(f"scenario must be one of {ALL_SCENARIOS}, got {scenario!r}")
