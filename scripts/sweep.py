"""Reading the saved learning-rate sweep.
Called from `notebooks/Figure2.ipynb` and `notebooks/Figure6.ipynb`.

Figures 2 and 6 rest on one sweep, too long to re-run for a figure:
50 magnitudes x 50 discriminabilities x 20 expansion rates x 20 contraction
rates, each cell a bandit task averaged over its agents.

Rather than ship that file, `data/` holds four aggregates of it.

    sweep_by_rates.csv           400 rows, one per (alpha_exp, alpha_con),
                                 averaged over all 2500 environments
    sweep_epistemic_by_rates.csv the same grid, read through the epistemic
                                 column, for Fig. 6B
    sweep_by_environment.csv     2500 rows, one per (magnitude,
                                 discriminability), averaged over the rate
                                 triangle
    sweep_groups.npz             the per-agent vectors averaged element-wise
                                 within each level: their mean is the curve of
                                 Fig. 2C and 2E, their SD the band
The "triangle" is the half of the rate square where contraction is slower than
expansion, i.e. where alpha_con <= alpha_exp. 
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The sweep data is stored in `data/` as a set of CSVs and NPZs. The CSVs are
# the aggregates that are used for plotting, and the NPZs are the full sweep
__all__ = ["DATA_DIR", "load_cells", "load_groups", "load_randoma_groups",
           "load_rates",
           "load_epistemic_rates",
           "load_environments", "load_discriminability", "load_magnitude",
           "triangle", "top_cells"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_cells(folder=None):
    """Every record of the sweep, one row per (environment, rate pair).

    475,000 rows: the per-agent mean and SD of each model's accuracy, plus the
    epistemic criterion. 6 MB, against the 3.1 GB of the JSON it came from.
    """
    return dict(np.load((folder or DATA_DIR) / "sweep_cells.npz"))


def load_groups(folder=None):
    """The per-agent vectors, element-wise averaged within each level.

    This is what `analysis_figure_3A_3B_3C.py` plots: for one magnitude or one
    discriminability level, the 200-agent vector averaged element-wise over
    every cell of that level. Its mean is the curve and its SD is the shaded
    band — a spread **across agents**, which is much narrower than the spread
    across learning-rate cells.
    """
    return dict(np.load((folder or DATA_DIR) / "sweep_groups.npz"))


def load_randoma_groups(folder=None):
    """The same group vectors, from the random-learning-rate run.

    Panels C and E of the published Fig. 2 come from
    `50m_50d_randoma_softmax_bandits_data.json`, not from the exhaustive
    sweep: 100 agents per cell, each with its own learning rates drawn at
    random. That is where their shaded band comes from, and why it is visible
    at all. On the sweep the same calculation gives a band of 0.002, because
    averaging 190 rate cells element-wise collapses the spread across agents.
    """
    return dict(np.load((folder or DATA_DIR) / "randoma_groups.npz"))


def _read(name: str, folder: Path | None = None) -> pd.DataFrame:
    path = (folder or DATA_DIR) / f"sweep_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/aggregate_sweep.py on the full "
            f"sweep to regenerate the aggregates.")
    return pd.read_csv(path)


def load_rates(folder=None):
    """Accuracy per (alpha_exp, alpha_con), as two square matrices.

    Returns ``(alpha_exp_values, alpha_con_values, abs_matrix, ra_matrix)``,
    the matrices indexed [expansion, contraction].
    """
    table = _read("by_rates", folder)
    ra = table.pivot(index="alpha_exp", columns="alpha_con", values="acc_ra")
    ab = table.pivot(index="alpha_exp", columns="alpha_con", values="acc_abs")
    return (ra.index.to_numpy(), ra.columns.to_numpy(),
            ab.to_numpy(), ra.to_numpy())


def load_epistemic_rates(folder=None):
    """Epistemic accuracy per (alpha_exp, alpha_con), for Fig. 6B.

    The same sweep as the instrumental panel, read through its `epi_acc_diff`
    column, so the two panels of Figure 6 share one 20 x 20 rate grid and one
    set of environments. Returns
    ``(alpha_exp_values, alpha_con_values, epistemic, instrumental)`` — the
    instrumental accuracy of that same run comes along so that the inset of
    panel B can be read off the same cells.
    """
    table = _read("epistemic_by_rates", folder)
    grid = table.pivot(index="alpha_exp", columns="alpha_con",
                       values="epistemic")
    ra = table.pivot(index="alpha_exp", columns="alpha_con", values="acc_ra")
    return (grid.index.to_numpy(), grid.columns.to_numpy(), grid.to_numpy(),
            ra.to_numpy())


def load_environments(folder=None):
    """Accuracy per (magnitude, discriminability), as two matrices.

    Returns ``(magnitudes, discriminabilities, abs_matrix, ra_matrix)``.
    """
    table = _read("by_environment", folder)
    ra = table.pivot(index="magnitude", columns="discriminability",
                     values="acc_ra")
    ab = table.pivot(index="magnitude", columns="discriminability",
                     values="acc_abs")
    return (ra.index.to_numpy(), ra.columns.to_numpy(),
            ab.to_numpy(), ra.to_numpy())


def load_discriminability(folder=None):
    """Mean and SD of accuracy per discriminability level."""
    return _read("by_discriminability", folder)


def load_magnitude(folder=None):
    """Mean accuracy per magnitude level."""
    return _read("by_magnitude", folder)


def triangle(matrix, exp_values, con_values):
    """Blank out the cells where contraction would outpace expansion."""
    keep = con_values[None, :] <= exp_values[:, None]
    return np.where(keep, matrix, np.nan)


def top_cells(score, fraction=0.05):
    """Mask of the best `fraction` of the cells that exist.

    The count is taken over the cells actually shown, not over the padded
    square, so that 5% of the triangle means 5% of its 210 combinations.
    """
    valid = ~np.isnan(score)
    k = max(1, int(round(valid.sum() * fraction)))
    threshold = np.sort(score[valid])[-k]
    return valid & (score >= threshold)
