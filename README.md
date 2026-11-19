# Achieving Scale-Invariant Reinforcement Learning Performance with Reward Range Normalization

Code and aggregated data reproducing figures and analyses of the article.

    pip install -r requirements.txt

## What is here

    normnorm/      the models and the tasks
                     bandit.py        two-armed bandit, ABS and RA
                     dynamic.py       expanding, shrinking and drifting ranges
                     narmed.py        N-armed and non-stationary bandits
                     gridworld.py     the 4x4 grid
                     seasons.py       the seasonal gridworld
                     sweep.py         readers for the learning-rate sweep
                     optimal_rates.py the expansion/contraction grid
                     tradeoff.py      trade-off quality and apparent beta
    data/          the aggregates the figures read, about a megabyte
    notebooks/     Figure1.ipynb to Figure7.ipynb (not part of this release)
    style.py       colours, sizes and the save helper, shared by the notebooks (not part of this release)

## Running a new simulation

The notebooks redraw the published figures from the aggregates. To simulate
something new, import from `normnorm` directly.

Example for a stationary two-armed bandit, 200 agents at one magnitude and one
discriminability:

```python
import numpy as np
from normnorm.bandit import make_rewards, simulate_bandit

rng = np.random.default_rng(0)
rewards = make_rewards(magnitude=10.0, discriminability=0.25, n_trials=80)

alpha_q = rng.random(200)                  # one Q learning rate per agent
alpha_exp = rng.random(200)                # expansion of the range
alpha_con = alpha_exp * rng.random(200)    # contraction, always the slower

abs_ = simulate_bandit(rewards, "ABS", alpha_q, beta=1.0)
ra = simulate_bandit(rewards, "RA", alpha_q, beta=1.0,
                     alpha_exp=alpha_exp, alpha_con=alpha_con)
print(abs_.correct.mean(), ra.correct.mean())
```