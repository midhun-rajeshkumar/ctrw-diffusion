"""Mean-squared displacement and anomalous-diffusion exponent estimation.

For a continuous-time random walk the ensemble mean-squared displacement (MSD)
grows as a power of time,

    MSD(t) = mean over particles of x(t)**2  ~  t ** alpha,

where the exponent ``alpha`` characterises the diffusion regime: ``alpha == 1``
is ordinary (Fickian) diffusion, ``alpha < 1`` is subdiffusion and ``alpha > 1``
is superdiffusion.

This module computes the MSD from an ensemble of trajectories and estimates
``alpha`` by a straight-line fit to the data in log-log coordinates, where the
power law becomes ``log(MSD) = alpha * log(t) + intercept``. The quality of that
straight-line fit is reported as the coefficient of determination ``r_squared``.

All functions here are deterministic and are tested against hand-computed
examples; no random numbers are drawn in this module.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalingFit:
    """Result of fitting a power law to a mean-squared-displacement curve.

    Attributes
    ----------
    exponent : float
        The estimated anomalous-diffusion exponent ``alpha``.
    intercept : float
        The intercept of the log-log straight-line fit.
    r_squared : float
        Coefficient of determination of the log-log fit, between 0 and 1; values
        close to 1 indicate a clean power law.
    """

    exponent: float
    intercept: float
    r_squared: float


@dataclass
class AnalysisResult:
    """Bundle of the MSD curve and its power-law fit, ready for reporting.

    Attributes
    ----------
    time_grid : numpy.ndarray
        Observation times at which the MSD was evaluated.
    msd : numpy.ndarray
        Mean-squared displacement at each observation time.
    fit : ScalingFit
        The fitted exponent, intercept and goodness of fit.
    """

    time_grid: np.ndarray
    msd: np.ndarray
    fit: ScalingFit


def mean_squared_displacement(ensemble):
    """Return the ensemble mean-squared displacement at each observation time."""
    ensemble = np.asarray(ensemble, dtype=float)
    return np.mean(ensemble ** 2, axis=0)


def fit_scaling_exponent(time_grid, msd):
    """Fit a power law to an MSD curve in log-log coordinates."""
    time_grid = np.asarray(time_grid, dtype=float)
    msd = np.asarray(msd, dtype=float)
    valid = (time_grid > 0) & (msd > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "need at least two points with positive time and MSD to fit a "
            "power law, got {}".format(int(np.count_nonzero(valid)))
        )
    log_t = np.log(time_grid[valid])
    log_msd = np.log(msd[valid])
    exponent, intercept = np.polyfit(log_t, log_msd, 1)
    # Coefficient of determination of the straight-line fit in log-log space.
    predicted = exponent * log_t + intercept
    residual_ss = np.sum((log_msd - predicted) ** 2)
    total_ss = np.sum((log_msd - np.mean(log_msd)) ** 2)
    r_squared = 1.0 if total_ss == 0 else 1.0 - residual_ss / total_ss
    return ScalingFit(
        exponent=float(exponent),
        intercept=float(intercept),
        r_squared=float(r_squared),
    )


def analyze_ensemble(time_grid, ensemble):
    """Compute the MSD of an ensemble and fit its scaling exponent."""
    time_grid = np.asarray(time_grid, dtype=float)
    msd = mean_squared_displacement(ensemble)
    fit = fit_scaling_exponent(time_grid, msd)
    return AnalysisResult(time_grid=time_grid, msd=msd, fit=fit)
