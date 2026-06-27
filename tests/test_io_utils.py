"""Tests for the :mod:`ctrw_diffusion.io_utils` module.

Saving and loading are exercised on real files created inside pytest's
``tmp_path`` temporary directory, so the tests touch the file system without
leaving artifacts behind. The central property is the round trip: data written
and read back must be unchanged.
"""

import numpy as np

from ctrw_diffusion import io_utils
from ctrw_diffusion.analysis import AnalysisResult, ScalingFit


def test_ensemble_round_trip_preserves_arrays(tmp_path):
    """An ensemble saved and reloaded is unchanged.

    GIVEN: a time grid and a particle-by-time position array
    WHEN: they are saved and then loaded back
    THEN: both arrays match the originals
    """
    time_grid = np.linspace(0.0, 10.0, 6)
    ensemble = np.arange(18.0).reshape(3, 6)
    path = tmp_path / "ensemble.npz"

    io_utils.save_ensemble(path, time_grid, ensemble)
    loaded_grid, loaded_ensemble = io_utils.load_ensemble(path)

    assert np.allclose(loaded_grid, time_grid)
    assert np.allclose(loaded_ensemble, ensemble)


def test_analysis_round_trip_preserves_fit(tmp_path):
    """An analysis result saved and reloaded keeps its fitted values.

    GIVEN: an analysis result with a known exponent, intercept and R-squared
    WHEN: it is saved and then loaded back
    THEN: the reloaded fit carries the same values
    """
    fit = ScalingFit(exponent=1.25, intercept=-0.5, r_squared=0.987)
    time_grid = np.linspace(1.0, 5.0, 5)
    msd = time_grid ** 1.25
    result = AnalysisResult(time_grid=time_grid, msd=msd, fit=fit)
    path = tmp_path / "analysis.json"

    io_utils.save_analysis(path, result)
    loaded = io_utils.load_analysis(path)

    assert loaded.fit.exponent == fit.exponent
    assert loaded.fit.intercept == fit.intercept
    assert loaded.fit.r_squared == fit.r_squared


def test_analysis_round_trip_preserves_curve(tmp_path):
    """The MSD curve survives a save and load unchanged.

    GIVEN: an analysis result carrying a time grid and MSD curve
    WHEN: it is saved and then loaded back
    THEN: the time grid and MSD arrays match the originals
    """
    fit = ScalingFit(exponent=0.8, intercept=0.1, r_squared=0.95)
    time_grid = np.linspace(1.0, 9.0, 9)
    msd = time_grid ** 0.8
    result = AnalysisResult(time_grid=time_grid, msd=msd, fit=fit)
    path = tmp_path / "analysis.json"

    io_utils.save_analysis(path, result)
    loaded = io_utils.load_analysis(path)

    assert np.allclose(loaded.time_grid, time_grid)
    assert np.allclose(loaded.msd, msd)


def test_loaded_analysis_is_correct_type(tmp_path):
    """Loading reconstructs the proper result and fit types.

    GIVEN: a saved analysis result
    WHEN: it is loaded back
    THEN: the result is an AnalysisResult holding a ScalingFit
    """
    fit = ScalingFit(exponent=1.0, intercept=0.0, r_squared=1.0)
    time_grid = np.linspace(1.0, 4.0, 4)
    result = AnalysisResult(time_grid=time_grid, msd=time_grid, fit=fit)
    path = tmp_path / "analysis.json"

    io_utils.save_analysis(path, result)
    loaded = io_utils.load_analysis(path)

    assert isinstance(loaded, AnalysisResult)
    assert isinstance(loaded.fit, ScalingFit)
