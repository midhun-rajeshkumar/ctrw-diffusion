"""Tests for the :mod:`ctrw_diffusion.analysis` module.

The MSD is checked against a hand-computed ensemble. The power-law fit is checked
against data constructed to follow an exact power law, for which the exponent and
a coefficient of determination of one are known in advance. Floating-point
results are compared with tolerances rather than exact equality.
"""

import numpy as np
import pytest

from ctrw_diffusion import analysis


def test_msd_of_known_ensemble():
    """The MSD is the mean of squared positions over particles.

    GIVEN: two particles at positions (0, 2) and (0, 4) over two times
    WHEN: the mean-squared displacement is computed
    THEN: it equals (0, mean(4, 16)) = (0, 10)
    """
    ensemble = np.array([[0.0, 2.0], [0.0, 4.0]])
    result = analysis.mean_squared_displacement(ensemble)
    assert np.allclose(result, [0.0, 10.0])


def test_fit_recovers_known_exponent():
    """A clean power law is fitted with the correct exponent.

    GIVEN: msd = t**1.5 sampled on positive times
    WHEN: the scaling exponent is fitted
    THEN: the recovered exponent is 1.5
    """
    time_grid = np.linspace(1.0, 100.0, 50)
    msd = time_grid ** 1.5
    fit = analysis.fit_scaling_exponent(time_grid, msd)
    assert fit.exponent == pytest.approx(1.5, abs=1e-6)


def test_fit_of_perfect_power_law_has_unit_r_squared():
    """A perfect power law yields a coefficient of determination of one.

    GIVEN: msd = t**0.7, an exact power law
    WHEN: the scaling exponent is fitted
    THEN: the coefficient of determination is one
    """
    time_grid = np.linspace(1.0, 100.0, 50)
    msd = time_grid ** 0.7
    fit = analysis.fit_scaling_exponent(time_grid, msd)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_ignores_non_positive_times():
    """The undefined logarithm at t=0 does not break the fit.

    GIVEN: a grid that starts at zero with a matching zero MSD
    WHEN: the scaling exponent is fitted
    THEN: the fit succeeds and recovers the underlying exponent
    """
    time_grid = np.linspace(0.0, 100.0, 51)
    msd = np.empty_like(time_grid)
    msd[0] = 0.0
    msd[1:] = time_grid[1:] ** 1.2
    fit = analysis.fit_scaling_exponent(time_grid, msd)
    assert fit.exponent == pytest.approx(1.2, abs=1e-6)


def test_fit_requires_at_least_two_points():
    """Too few valid points is reported as an error.

    GIVEN: a single valid (time, msd) point
    WHEN: the scaling exponent is fitted
    THEN: a ValueError is raised
    """
    with pytest.raises(ValueError):
        analysis.fit_scaling_exponent([1.0], [3.0])


def test_analyze_ensemble_bundles_msd_and_fit():
    """The convenience wrapper returns the MSD curve and a fit together.

    GIVEN: an ensemble whose MSD follows a known power law in time
    WHEN: the ensemble is analysed
    THEN: the result carries the MSD array and a fit with the right exponent
    """
    time_grid = np.linspace(1.0, 100.0, 50)
    # Build an ensemble of one particle whose squared position is exactly t**1.0.
    ensemble = np.sqrt(time_grid).reshape(1, -1)
    result = analysis.analyze_ensemble(time_grid, ensemble)
    assert np.allclose(result.msd, time_grid)
    assert result.fit.exponent == pytest.approx(1.0, abs=1e-6)
