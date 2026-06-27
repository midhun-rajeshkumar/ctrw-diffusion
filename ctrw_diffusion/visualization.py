"""Plotting helpers for trajectories and mean-squared-displacement curves.

These functions perform no computation on the data: they receive arrays and
result objects that have already been produced by the tested modules and only
arrange them for display. Keeping preprocessing out of the plotting layer is why
this module needs no unit tests, while the scientific logic that does the real
work remains fully tested elsewhere.

Every function returns a :class:`matplotlib.figure.Figure` so the caller decides
whether to show it interactively or save it to a file.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(time_grid, ensemble, max_paths=20):
    """Plot a sample of individual particle trajectories against time.

    Parameters
    ----------
    time_grid : array_like
        Observation times.
    ensemble : array_like
        Array of shape ``(n_particles, n_times)`` of particle positions.
    max_paths : int, optional
        Maximum number of trajectories to draw, for legibility. Defaults to 20.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the trajectory plot.
    """
    ensemble = np.asarray(ensemble, dtype=float)
    n_shown = min(max_paths, ensemble.shape[0])
    figure, axis = plt.subplots(figsize=(7, 4))
    for row in ensemble[:n_shown]:
        axis.plot(time_grid, row, linewidth=0.8, alpha=0.7)
    axis.set_xlabel("time")
    axis.set_ylabel("position")
    axis.set_title("Sample of {} trajectories".format(n_shown))
    figure.tight_layout()
    return figure


def plot_msd(result):
    """Plot a mean-squared-displacement curve with its fitted power law.

    The data and the fitted straight line are shown on logarithmic axes, where a
    power law appears straight. The fitted exponent and goodness of fit are shown
    in the legend.

    Parameters
    ----------
    result : ctrw_diffusion.analysis.AnalysisResult
        The analysed ensemble, carrying the time grid, the MSD curve and the fit.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the MSD plot.
    """
    time_grid = np.asarray(result.time_grid, dtype=float)
    msd = np.asarray(result.msd, dtype=float)
    fit = result.fit
    positive = (time_grid > 0) & (msd > 0)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.loglog(time_grid[positive], msd[positive], "o", markersize=4, label="MSD")
    fitted = np.exp(fit.intercept) * time_grid[positive] ** fit.exponent
    axis.loglog(
        time_grid[positive],
        fitted,
        "-",
        label="fit: alpha={:.3f}, R^2={:.3f}".format(fit.exponent, fit.r_squared),
    )
    axis.set_xlabel("time")
    axis.set_ylabel("mean-squared displacement")
    axis.set_title("MSD scaling")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_analysis(time_grid, ensemble, result):
    """Plot trajectories and the MSD fit side by side as one overview figure.

    Parameters
    ----------
    time_grid : array_like
        Observation times.
    ensemble : array_like
        Array of shape ``(n_particles, n_times)`` of particle positions.
    result : ctrw_diffusion.analysis.AnalysisResult
        The analysed ensemble carrying the MSD curve and the fit.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with the trajectories on the left and the MSD fit on the right.
    """
    ensemble = np.asarray(ensemble, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    msd = np.asarray(result.msd, dtype=float)
    fit = result.fit
    positive = (time_grid > 0) & (msd > 0)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    n_shown = min(20, ensemble.shape[0])
    for row in ensemble[:n_shown]:
        left.plot(time_grid, row, linewidth=0.8, alpha=0.7)
    left.set_xlabel("time")
    left.set_ylabel("position")
    left.set_title("Sample of {} trajectories".format(n_shown))

    right.loglog(time_grid[positive], msd[positive], "o", markersize=4, label="MSD")
    fitted = np.exp(fit.intercept) * time_grid[positive] ** fit.exponent
    right.loglog(
        time_grid[positive],
        fitted,
        "-",
        label="fit: alpha={:.3f}, R^2={:.3f}".format(fit.exponent, fit.r_squared),
    )
    right.set_xlabel("time")
    right.set_ylabel("mean-squared displacement")
    right.set_title("MSD scaling")
    right.legend()

    figure.tight_layout()
    return figure
