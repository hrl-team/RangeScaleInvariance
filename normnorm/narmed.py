"""N-armed bandit tasks and learners for Fig. 5.
Called from 'notebooks/Figure5.ipynb'.

The published simulations were run through the COBA library (COntextual Bandit
Algorithms, v8.0.3), which generated the environments and drove the agents
through them. COBA is a large dependency for what it does here (iterate
learners over reward trajectories) so this module reproduces the same thing in
NumPy.

Tasks
-----
Each environment draws its own number of arms (2, 3 or 4) and its own reward
magnitude m, log-uniform over twelve orders of magnitude. Arm rewards then move
according to the task:

    fixed         drawn uniformly on [0, m], redrawn until the two closest arms
                  differ by more than 0.1 * m, then held constant
    noisy_fixed   the same, plus Gaussian noise whose SD is drawn between half
                  and all of the smallest gap between arms
    rw            a random walk per arm, steps of SD 0.1 * m, cumulated
    sin           sin(t / d + a) * m, with a phase shift per arm and a period
                  drawn per environment

Learners
--------
Feedback is partial: only the chosen arm's reward is seen, so the range and the
running statistics are updated from that reward alone.

    Random        uniform choice, the floor of the figure
    ABS softmax   delta = r - Q[a]
    RA softmax    delta = (r - Rmin) / (Rmax - Rmin) - Q[a]
"""

from __future__ import annotations

import numpy as np

__all__ = ["TRAJECTORY_TYPES", "make_environment", "make_environments",
           "FAMILIES", "draw_learner_params", "run_environment"]

TRAJECTORY_TYPES = ("fixed", "noisy_fixed", "rw", "sin")
ALL_TRAJECTORY_TYPES = TRAJECTORY_TYPES + ("fluctuating_magnitude",)

FAMILIES = ("Random", "AbsoluteSoftmax", "RangeSoftmax")

MAGNITUDES = np.logspace(-6, 6, 1000)
N_INTERACTIONS = 80


# --------------------------------------------------------------------------- #
# Reward trajectories
# --------------------------------------------------------------------------- #

def _fixed_rewards(n_actions, m):
    """Arm rewards far enough apart to be distinguishable, then held constant.

    Rejection sampling: redraw until every pair of adjacent
    arms differs by more than a tenth of the magnitude.
    """
    resolution = 0.1 * m
    while True:
        rewards = np.sort(np.random.uniform(0.0, m, size=n_actions))
        if np.all(np.diff(rewards) > resolution):
            return rewards


def _inject_noise(trajectory):
    """Gaussian noise whose SD sits between half and all of the smallest gap."""
    gaps = np.diff(np.sort(trajectory, axis=0), axis=0)
    min_gap = gaps.min()
    sd = np.random.uniform(min_gap / 2, min_gap)
    return trajectory + np.random.normal(0.0, sd, size=trajectory.shape)


def _fluctuate_magnitude(trajectory):
    """Rescale by a fresh magnitude every ten trials, shuffling the arms."""
    out = np.zeros_like(trajectory)
    for block in range(trajectory.shape[1] // 10):
        m = np.random.choice(np.logspace(-6, 6, 100))
        sl = slice(block * 10, block * 10 + 10)
        out[:, sl] = trajectory[:, sl] * m
        np.random.shuffle(out[:, sl])
    return out


def make_environment(seed, trajectory_type, n_interactions=N_INTERACTIONS):
    """One environment, fully determined by its seed and its task.

    Returns a dict with the (n_actions, n_interactions) reward matrix, the
    number of arms, the magnitude, and the per-trial best and worst rewards.
    """
    if trajectory_type not in ALL_TRAJECTORY_TYPES:
        raise ValueError(f"unknown trajectory type {trajectory_type!r}")

    np.random.seed(seed)
    n_actions = int(np.random.choice([2, 3, 4]))
    m = float(np.random.choice(MAGNITUDES))

    if trajectory_type == "rw":
        steps = np.random.normal(0.0, m * 0.1, size=(n_actions, n_interactions))
        trajectory = np.cumsum(steps, axis=1)
    elif trajectory_type == "sin":
        period = np.random.uniform(n_interactions / 10, n_interactions / 2)
        t = np.arange(n_interactions)
        trajectory = np.sin(t / period + np.arange(n_actions)[:, None]) * m
    else:
        trajectory = np.repeat(_fixed_rewards(n_actions, m)[:, None],
                               n_interactions, axis=1)
        if trajectory_type == "noisy_fixed":
            trajectory = _inject_noise(trajectory)
        elif trajectory_type == "fluctuating_magnitude":
            trajectory = _fluctuate_magnitude(trajectory)

    return {"rewards": trajectory, "n_actions": n_actions, "magnitude": m,
            "seed": seed, "trajectory_type": trajectory_type,
            "max_reward": trajectory.max(axis=0),
            "min_reward": trajectory.min(axis=0),
            "correct_action": trajectory.argmax(axis=0)}


def make_environments(n_seeds, trajectory_types=TRAJECTORY_TYPES,
                      n_interactions=N_INTERACTIONS):
    """One environment per (seed, task) pair."""
    return [make_environment(seed, tt, n_interactions)
            for seed in range(n_seeds) for tt in trajectory_types]


# --------------------------------------------------------------------------- #
# Learners
# --------------------------------------------------------------------------- #

def draw_learner_params(n_learners, rng):
    """Learning rates and seeds for one family.
    """
    aexp = rng.uniform(0.1, 0.9, n_learners)
    return {"seed": rng.integers(0, 10000, n_learners),
            "alpha": rng.uniform(0.1, 0.9, n_learners),
            "alpha_Z": rng.uniform(0.1, 0.9, n_learners), # was used for the z-score model, but not relevant in the main manuscript
            "aexp": aexp,
            "acon": np.array([rng.uniform(0.1, a) for a in aexp])}


def _softmax(q, beta):
    exponent = np.clip(q * beta, None, 709.0)
    exp = np.exp(exponent - exponent.max())
    return exp / exp.sum()


def run_environment(env, family, params, index, beta=1.0):
    """Run one agent through one environment.

    Returns the total reward collected, and the proportion of trials on which
    the best arm was chosen.
    """
    rewards = env["rewards"]
    n_actions, n_trials = rewards.shape
    rng = np.random.default_rng(int(params["seed"][index]))

    q = np.zeros(n_actions)
    alpha = float(params["alpha"][index])
    total = 0.0
    n_correct = 0

    if family == "RangeSoftmax":
        r_min = r_max = 0.0
        aexp, acon = float(params["aexp"][index]), float(params["acon"][index])

    for t in range(n_trials):
        if family == "Random":
            action = rng.integers(n_actions)
        else:
            action = rng.choice(n_actions, p=_softmax(q, beta))

        reward = rewards[action, t]
        total += reward
        n_correct += action == env["correct_action"][t]

        if family == "Random":
            continue

        if family == "AbsoluteSoftmax":
            target = reward
        elif family == "RangeSoftmax":
            min_pe, max_pe = reward - r_min, reward - r_max
            r_min += (aexp if min_pe < 0 else acon) * min_pe
            r_max += (aexp if max_pe > 0 else acon) * max_pe
            target = ((reward - r_min) / (r_max - r_min)
                      if r_max != r_min else reward)

        q[action] += alpha * (target - q[action])

    return total, n_correct / n_trials
