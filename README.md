# Achieving scale-independent reinforcement learning performance with reward range normalization

## Table of Contents
1. [Overview](#overview)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)
7. [Fundings](#fundings)

---

## Overview

The repository contains the codes to reproduce the simulations presented in the preprint available here: https://doi.org/10.31234/osf.io/bjyr9. 

Key Contributions:
Range Adaptation Mechanism: Normalizes reward encoding within a fixed scale, yielding stable performance across various conditions.
Model Comparison: Demonstrates RA’s scale-invariant performance in two-armed bandit tasks and multi-step decision scenarios.
Exploration-Exploitation Balance: Shows RA’s capability to optimize exploration automatically.

This repository includes scripts and code to replicate the simulations and analyses outlined in the article.


## Requirements

- [python](https://www.python.org/downloads/release/python-31012/) 3.10.12
- [numpy](https://numpy.org/) v.1.25.2
- [pandas](https://pandas.pydata.org/) v.2.1.0
- [seaborn](https://seaborn.pydata.org/) v.0.13.0
- [matplotlib](https://matplotlib.org/) v.3.8.0
- [hyperopt](https://github.com/hyperopt/hyperopt) v.0.2.7 for parameter optimization
- [progressbar](https://pypi.org/project/progressbar2/) v.2.5 for tracking progress in simulations
- [COBA](https://coba-docs.readthedocs.io/en/latest/getting_started.html) v.8.0.5 for multi-tasks simulations
- [tqdm](https://tqdm.github.io/) v.4.66.1 for tracking progress in COBA simulations
- [scipy](https://scipy.org/) v.1.11.2


## Usage

This section provides a recommended order to explore the codebase, starting with foundational modules and moving to analyses associated with figures from the paper.

#### Foundational Scripts
To understand the core functions and modules used across all analyses:

1. **`agents.py`** - Defines agent behaviors and learning strategies.
2. **`models.py`** - Establishes core models and simulation frameworks.
3. **`general_functions.py`** - Contains functions used across different analyses.

#### Analysis by Figures in the Article

Scripts are organized by figure for replicating analyses and visualizations from the paper:

- **Figure 3: Performance in two-armed bandits tasks**:
  - `simulations_figure3.py` - Runs simulations for ABS and RA agents in two-armed bandits tasks with varying magnitude and discriminability levels.
  - `analysis_figure3A_3B_3C.py` - Analysis scripts for parts A, B, and C of Figure 3: Bandit task performance.
  - `analysis_figure3D.py` - Simulations and analysis script specific to exploring the trade-off quality per ABS β value and magnitude level and estimating RA's apparent β value.

- **Figure 4: Behavioral signatures of the ABS and RA models**:
  - `analysis_figure4.py` - Performs simulations and analyses for comparing performance signatures of ABS and RA model at the population level, and to show what happens when the β value is manipulated.

- **Figure 5: RA model latent variables**:
  - `analysis_figure5.py` - Code for simulations and analysis specific to RA model latent variables.

## License

This project is licensed under the [...] License. 

## Acknowledgments

We would like to thank Julien Karadayi for his valuable contributions during the code review process, providing insightful feedback that enhanced the organization and readability of this code.
Additionally, we wish to acknowledge the invaluable resources provided by the open-source libraries and tools used in this project.

## Fundings
This research project received the support from the European Research Council consolidator grant (RaReMem:  101043804) and three Agence Nationale de  la Recherche grants (CogFinAgent: ANR-21-CE23-0002-02; RELATIVE: ANR-21-CE37-750  0008-01; RANGE: ANR-21-CE28-0024-01), the Alexander Von Humbolt foundation and a Google unrestricted gift.

