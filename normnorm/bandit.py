"""Two-armed bandit task and the two learning rules.

Called in 'notebooks/Figure3.ipynb' and 'notebooks/Figure4.ipynb'

The two models differ only in what they feed to the prediction error:

    ABS   delta = r - Q[chosen]                       (absolute reward)
    RA    delta = s(r) - Q[chosen],  s(r) = (r - Rmin) / (Rmax - Rmin)

where Rmin and Rmax are running estimates of the context's reward range,
updated asymmetrically: expansions of the range are applied with learning rate
alpha_exp, contractions with alpha_con, and alpha_con < alpha_exp.

Everything is vectorised over agents: one call simulates n_agents independent
learners on the same reward sequence
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_rewards", "simulate_bandit", "BanditResult"] # 


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

def make_rewards(magnitude: float, discriminability: float, n_trials: int) -> np.ndarray:
    """Deterministic rewards of a two-armed bandit.

    The two options are placed symmetrically around 0.5 and then scaled:

        worst = (0.5 - d/2) * m        best = (0.5 + d/2) * m

    so that `discriminability` controls how far apart the options are and
    `magnitude` sets the overall scale. With d = 1 the worst option pays
    nothing. Option 1 is always the better one.

    Returns
    -------
    (n_trials, 2) array, constant across trials.
    """
    rewards = np.empty((n_trials, 2))
    rewards[:, 0] = (0.5 - discriminability / 2) * magnitude
    rewards[:, 1] = (0.5 + discriminability / 2) * magnitude
    return rewards


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

class BanditResult:
    """Per-agent traces returned by :func:`simulate_bandit`.

    Attributes
    ----------
    correct : (n_agents, n_trials) bool
        Whether the better option was chosen.
    q : (n_agents, n_trials + 1, 2)
        Q-value traces, including the initial values.
    rmin, rmax : (n_agents, n_trials + 1) or None
        Range estimates; ``None`` for the absolute model.
    """

    def __init__(self, correct, q, rmin=None, rmax=None):
        self.correct = correct # boolean per agent and trial
        self.q = q # the two Q-values per agent and trial, including the initial values
        self.rmin = rmin # the running minimum per agent and trial, including the initial values
        self.rmax = rmax # the running maximum per agent and trial, including the initial values

    @property # to make it read-only
    def accuracy(self) -> np.ndarray:
        """Proportion of correct choices, one value per agent."""
        return self.correct.mean(axis=1)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def simulate_bandit(
    rewards: np.ndarray,
    model: str,
    alpha_q: np.ndarray,
    beta: float = 1.0,
    alpha_exp: np.ndarray | None = None,
    alpha_con: np.ndarray | None = None,
    q_init: float = 0.0,
    rng: np.random.Generator | None = None,
) -> BanditResult:
    """Run `n_agents` learners on one reward sequence.

    Parameters
    ----------
    rewards : (n_trials, 2)
        Output of :func:`make_rewards`.
    model : {"ABS", "RA"}
    alpha_q : (n_agents,)
        Q-value learning rate, one per agent.
    beta : float
        Inverse temperature of the softmax, shared by all agents.
    alpha_exp, alpha_con : (n_agents,)
        Range learning rates, required for ``model="RA"``.
    q_init : float
        Initial Q-value. 0 is used throughout
    rng : numpy Generator, optional
        Pass a seeded generator for reproducible runs.
    """
    if model not in ("ABS", "RA"):
        raise ValueError(f"model must be 'ABS' or 'RA', got {model!r}")
    if model == "RA" and (alpha_exp is None or alpha_con is None):
        raise ValueError("model='RA' requires alpha_exp and alpha_con")

    rng = np.random.default_rng() if rng is None else rng

    alpha_q = np.atleast_1d(np.asarray(alpha_q, dtype=float))
    n_agents = alpha_q.size
    n_trials = rewards.shape[0]

    q = np.full((n_agents, 2), float(q_init))
    q_trace = np.empty((n_agents, n_trials + 1, 2))
    q_trace[:, 0] = q

    if model == "RA":
        alpha_exp = np.atleast_1d(np.asarray(alpha_exp, dtype=float))
        alpha_con = np.atleast_1d(np.asarray(alpha_con, dtype=float))
        rmin = np.zeros(n_agents)
        rmax = np.zeros(n_agents)
        rmin_trace = np.empty((n_agents, n_trials + 1))
        rmax_trace = np.empty((n_agents, n_trials + 1))
        rmin_trace[:, 0] = rmin # the running minimum of the reward range for each agent
        rmax_trace[:, 0] = rmax # the running maximum of the reward range for each agent

    correct = np.empty((n_agents, n_trials), dtype=bool)
    best = int(np.argmax(rewards[0]))          # option 1 by construction
    agents = np.arange(n_agents)

    for t in range(n_trials):
        # --- choice -------------------------------------------------------- #
        if t == 0:
            # No information yet: pick at random, as in the original model.
            choice = rng.integers(0, 2, size=n_agents)
        else:
            # P(choose option 1). The exponent is clipped at 709, just below
            # where float64 exp() overflows. beyond that the softmax is
            # saturated anyway and the probability is 0 to machine precision.
            exponent = np.clip((q[:, 0] - q[:, 1]) * beta, None, 709.0)
            p_one = 1.0 / (1.0 + np.exp(exponent)) # probability of choosing option 1
            # if the probability of choosing option 1 is greater than a random number,
            # choose option 1, otherwise choose option 0
            choice = (p_one >= rng.random(n_agents)).astype(int)

        correct[:, t] = choice == best
        reward = rewards[t, choice]

        # --- learning ------------------------------------------------------ #
        if model == "ABS":
            target = reward
        else:
            # Range update.
            d_max = reward - rmax # the difference between the received reward and the current maximum reward estimate
            d_min = reward - rmin # the difference between the received reward and the current minimum reward estimate
            rmax += np.where(d_max > 0, alpha_exp, alpha_con) * d_max # update the running maximum of the reward range for each agent
            rmin += np.where(d_min < 0, alpha_exp, alpha_con) * d_min # update the running minimum of the reward range for each agent
            span = rmax - rmin
            # A degenerate range (first trial, or both bounds equal) leaves the
            # reward untransformed rather than dividing by zero.
            target = np.where(span != 0, (reward - rmin) / np.where(span != 0, span, 1.0), reward)
            rmin_trace[:, t + 1] = rmin
            rmax_trace[:, t + 1] = rmax

        # update the Q-value of the chosen option for each agent, similar for ABS and RA models
        q[agents, choice] += alpha_q * (target - q[agents, choice])
        q_trace[:, t + 1] = q

    if model == "ABS":
        return BanditResult(correct, q_trace)
    return BanditResult(correct, q_trace, rmin_trace, rmax_trace)
