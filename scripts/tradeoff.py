"""The exploration-exploitation trade-off of Fig. 2F.

Called in 'notebooks/Figure2.ipynb'.

Two quantities are computed here: 
- the quality of the trade-off for a fixed temperature (variable quality), and
- the apparent temperature of a range-adapted agent, as seen by an absolute agent (variable apparent_beta).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .bandit import make_rewards, simulate_bandit

__all__ = ["BETAS", "MAGNITUDES", "tradeoff_quality", "apparent_beta",
           "load_fits"]

# The folder containing the saved fits of the trade-off quality and apparent beta.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_fits(folder=None):
    """The apparent and best temperatures as they were fitted for the paper.

    500 magnitudes, searched with hyperopt over beta in [1e-5, 100] with 3000
    evaluations each.
    """
    return dict(np.load((folder or DATA_DIR) / "tradeoff_fits.npz"))

BETAS = np.linspace(0.1, 100, 500)
MAGNITUDES = np.logspace(-1, 1, 500)

REWARDS = (0.25, 0.75)       # a discriminability of 0.5, before scaling
DISCRIMINABILITY = 0.5
ALPHA_Q = 0.3                # the learning rate entering the closed form
C_FACTOR = 3                 # scaling of the quality score

# Parameters of the agents simulated for the apparent beta.
N_AGENTS = 100
N_TRIALS = 80
ALPHA_Q_ABS = 0.3
ALPHA_Q_RA = 0.3
ALPHA_EXP = 0.6493810914202369
ALPHA_CON = 0.5763100496046456

def tradeoff_quality(betas=BETAS, magnitudes=MAGNITUDES, rewards=REWARDS,
                     alpha_q=ALPHA_Q, c_factor=C_FACTOR):
    """Quality of the trade-off for every (beta, magnitude) pair.
    For details on the metric, see the paper Methods section:
    "Quantification and statistical analysis"
    "1.	Exploration-exploitation trade off quality metric"
    Returns a (len(betas), len(magnitudes)) array; brighter is better.
    """
    b = np.asarray(betas)[:, None]
    m = np.asarray(magnitudes)[None, :]
    leave_bad = 1.0 / (1.0 + np.exp(b * alpha_q * m * rewards[0]))
    leave_good = 1.0 / (1.0 + np.exp(b * m * (rewards[1] - rewards[0])))
    return c_factor * (leave_bad - leave_good)


def _delta_q(model, magnitude, beta, rng, **rates):
    """Trial-by-trial value difference of one model, (n_agents, n_trials).
    """
    rewards = make_rewards(magnitude, DISCRIMINABILITY, N_TRIALS)
    ones = np.ones(N_AGENTS)
    result = simulate_bandit(rewards, model, rates.pop("alpha_q") * ones,
                             beta=beta, rng=rng,
                             **{k: v * ones for k, v in rates.items()})
    q = result.q[:, :-1]                      # values entering each choice
    return q[:, :, 0] - q[:, :, 1]


def apparent_beta(magnitudes=MAGNITUDES, betas=None, n_refine=2, seed=0):
    """The absolute temperature that best imitates range adaptation.
    For details on the metric, see the paper Methods section:
    "Quantification and statistical analysis"
    "2.	Approach for estimating β_APPARENT""
    For each magnitude, scans `betas`, then refines twice around the best value.

    Returns
    -------
    (magnitudes,) array of apparent beta values.
    """
    if betas is None:
        betas = np.linspace(0.01, 100.0, 200)
    out = np.empty(len(magnitudes))

    for i, m in enumerate(magnitudes):
        # Range adaptation does not depend on the searched beta: its own
        # temperature is fixed at 1, so it is simulated once.
        delta_ra = _delta_q("RA", m, 1.0, np.random.default_rng(seed),
                            alpha_q=ALPHA_Q_RA, alpha_exp=ALPHA_EXP,
                            alpha_con=ALPHA_CON)

        grid = np.asarray(betas, dtype=float)
        for step in range(n_refine + 1):
            losses = np.empty(grid.size)
            for j, beta in enumerate(grid):
                delta_abs = _delta_q("ABS", m, beta,
                                     np.random.default_rng(seed),
                                     alpha_q=ALPHA_Q_ABS)
                losses[j] = np.sum((delta_ra - beta * delta_abs) ** 2)
            best = int(np.argmin(losses))
            if step == n_refine:
                break
            # Refine inside the bracket around the current best.
            lo = grid[max(best - 1, 0)]
            hi = grid[min(best + 1, grid.size - 1)]
            grid = np.linspace(lo, hi, 12)

        out[i] = grid[best]
        print(f"  magnitude {m:.3g}: apparent beta = {out[i]:.2f}", flush=True)

    return out
