"""Persistence of trajectory ensembles and analysis results.

Saving and loading lets the three pipeline stages run independently: an ensemble
can be simulated once and written to disk, then analysed and plotted later
without repeating the simulation. This separation is what makes a stochastic
study reproducible and cheap to re-examine.

Two formats are used, each chosen to fit its data:

- ensembles are stored as compressed NumPy ``.npz`` archives holding the time
  grid and the particle-by-time position array;
- analysis results are stored as human-readable JSON so that the fitted exponent
  and its goodness of fit can be inspected or quoted directly.
"""

import json

import numpy as np

from .analysis import AnalysisResult, ScalingFit


def save_ensemble(path, time_grid, ensemble):
    """Write a trajectory ensemble and its time grid to a ``.npz`` archive."""
    time_grid = np.asarray(time_grid, dtype=float)
    ensemble = np.asarray(ensemble, dtype=float)
    np.savez_compressed(path, time_grid=time_grid, ensemble=ensemble)


def load_ensemble(path):
    """Read a trajectory ensemble previously saved with :func:`save_ensemble`."""
    with np.load(path) as data:
        return data["time_grid"], data["ensemble"]


def save_analysis(path, result):
    """Write an analysis result to a JSON file."""
    payload = {
        "time_grid": np.asarray(result.time_grid, dtype=float).tolist(),
        "msd": np.asarray(result.msd, dtype=float).tolist(),
        "fit": {
            "exponent": result.fit.exponent,
            "intercept": result.fit.intercept,
            "r_squared": result.fit.r_squared,
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_analysis(path):
    """Read an analysis result previously saved with :func:`save_analysis`."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fit = ScalingFit(
        exponent=payload["fit"]["exponent"],
        intercept=payload["fit"]["intercept"],
        r_squared=payload["fit"]["r_squared"],
    )
    return AnalysisResult(
        time_grid=np.asarray(payload["time_grid"], dtype=float),
        msd=np.asarray(payload["msd"], dtype=float),
        fit=fit,
    )
