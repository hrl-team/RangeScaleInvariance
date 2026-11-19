"""Which expansion and contraction rates are best, and best for what (Fig. 6).

Called in 'notebooks/Figure6.ipynb'.

The range-adapted model has two free parameters, and the question of Fig. 6 is
whether the same pair of values is optimal under two different criteria:

- Instrumental accuracy: how often the agent picks the better option
- Epistemic accuracy: how close the represented range ends up to the real one

"""

from __future__ import annotations

import numpy as np

from .bandit import make_rewards

__all__ = ["sweep_rates", "sweep_absolute", "top_fraction_mask"]


def _simulate(rewards, alpha_q, alpha_exp, alpha_con, beta, rng):
    """One environment, all (rate, agent) combinations at once.

    `alpha_q`, `alpha_exp` and `alpha_con` are flat arrays of equal length.
    `alpha_exp = None` runs the absolute model.

    Returns ``(accuracy, rmin, rmax)``; the two range arrays are the values
    reached at the end of the task, and are ``None`` for the absolute model.
    """
    n = alpha_q.size
    n_trials = rewards.shape[0]
    relative = alpha_exp is not None

    q = np.zeros((n, 2))
    rows = np.arange(n)
    n_correct = np.zeros(n)
    if relative:
        rmin = np.zeros(n)
        rmax = np.zeros(n)

    for t in range(n_trials):
        if t == 0:
            choice = rng.integers(0, 2, size=n) # first trial is random
        else:
            exponent = np.clip((q[:, 0] - q[:, 1]) * beta, None, 709.0)
            choice = (1.0 / (1.0 + np.exp(exponent)) >= rng.random(n)).astype(int)

        n_correct += choice == 1            # option 1 is the better one
        reward = rewards[t, choice]

        if relative:
            d_max, d_min = reward - rmax, reward - rmin
            rmax += np.where(d_max > 0, alpha_exp, alpha_con) * d_max
            rmin += np.where(d_min < 0, alpha_exp, alpha_con) * d_min
            span = rmax - rmin
            target = np.where(span != 0,
                              (reward - rmin) / np.where(span != 0, span, 1.0),
                              reward)
        else:
            target = reward

        q[rows, choice] += alpha_q * (target - q[rows, choice])

    accuracy = n_correct / n_trials
    return (accuracy, rmin, rmax) if relative else (accuracy, None, None)


def _epistemic_distance(rmin, rmax, true_min, true_max):
    """Deviation of the represented range from the true one, as a proportion.
    see in the methods section of the manuscript:
    "Quantification and statistical analysis"
    "3.	Epistemic accuracy metric"
    """
    span = true_max - true_min
    if span == 0:
        return np.zeros_like(rmax)
    return np.sqrt(((true_max - rmax) / span) ** 2 + ((rmin - true_min) / span) ** 2)


def sweep_rates(alpha_exps, alpha_cons, magnitudes, discriminabilities,
                n_agents=200, n_trials=80, beta=1.0, seed=0):
    """Both criteria over the expansion x contraction grid.

    Each cell is averaged over every environment (magnitude x discriminability)
    and every agent. Agents differ only in alpha_Q, drawn uniformly.

    Returns
    -------
    dict with ``instrumental`` and ``epistemic``, both of shape
    (len(alpha_exps), len(alpha_cons)). Epistemic accuracy is minus the
    epistemic distance, so larger is better in both cases.
    """
    rng = np.random.default_rng(seed)
    n_exp, n_con = len(alpha_exps), len(alpha_cons)

    # Flatten (alpha_exp, alpha_con, agent) into one axis.
    exp_flat = np.repeat(np.repeat(alpha_exps, n_con), n_agents)
    con_flat = np.repeat(np.tile(alpha_cons, n_exp), n_agents)
    alpha_q = rng.random(n_exp * n_con * n_agents)

    instr = np.zeros(n_exp * n_con)
    epist = np.zeros(n_exp * n_con)
    n_env = 0

    for m in magnitudes:
        for d in discriminabilities:
            rewards = make_rewards(m, d, n_trials)
            true_min, true_max = rewards[0, 0], rewards[0, 1]

            acc, rmin, rmax = _simulate(rewards, alpha_q, exp_flat, con_flat,
                                        beta, rng)
            dist = _epistemic_distance(rmin, rmax, true_min, true_max)

            instr += acc.reshape(n_exp * n_con, n_agents).mean(axis=1)
            epist += dist.reshape(n_exp * n_con, n_agents).mean(axis=1)
            n_env += 1
        print(f"  magnitude {m:.3g} done", flush=True)

    return {"instrumental": (instr / n_env).reshape(n_exp, n_con),
            "epistemic": (-epist / n_env).reshape(n_exp, n_con)}


def sweep_absolute(magnitudes, discriminabilities, n_agents=200, n_trials=80,
                   beta=1.0, seed=0):
    """Instrumental accuracy of the absolute model,
    averaged over every environment and agent.
    """
    rng = np.random.default_rng(seed)
    alpha_q = rng.random(n_agents)
    total = np.zeros(n_agents)
    n_env = 0

    for m in magnitudes:
        for d in discriminabilities:
            rewards = make_rewards(m, d, n_trials)
            acc, _, _ = _simulate(rewards, alpha_q, None, None, beta, rng)
            total += acc
            n_env += 1

    return total / n_env


def top_fraction_mask(score, fraction=0.05):
    """Boolean mask of the `fraction` best cells of a score matrix."""
    k = max(1, int(round(score.size * fraction)))
    threshold = np.sort(score, axis=None)[-k]
    return score >= threshold
