"""Tests for the :mod:`ctrw_diffusion.simulation` module.

The deterministic core functions are checked against small hand-computed
examples. Floating-point results are compared with :func:`numpy.allclose` rather
than exact equality. Functions that draw random numbers are exercised with a
seeded generator and checked for shape, reproducibility and physical sanity.
"""

import numpy as np
import pytest

from ctrw_diffusion import simulation


def make_generator(seed=2024):
    """Return a freshly seeded generator for deterministic tests."""
    return np.random.default_rng(seed)


def test_event_times_are_cumulative():
    """Event times accumulate the waiting times.

    GIVEN: a short sequence of waiting times
    WHEN: the event times are computed
    THEN: they equal the running total of the waiting times
    """
    waiting = [1.0, 2.0, 0.5]
    result = simulation.event_times(waiting)
    assert np.allclose(result, [1.0, 3.0, 3.5])


def test_positions_accumulate_jumps_from_origin():
    """Positions accumulate the jumps starting from the origin.

    GIVEN: a short sequence of signed jumps
    WHEN: the positions after each event are computed
    THEN: they equal the running total of the jumps
    """
    jumps = [1.0, -3.0, 2.0]
    result = simulation.positions_at_events(jumps)
    assert np.allclose(result, [1.0, -2.0, 0.0])


def test_sample_on_grid_holds_last_position():
    """Between jumps the sampled position stays constant.

    GIVEN: events at t=1 and t=3 reaching positions 5 and 9
    WHEN: the trajectory is sampled at t = 0, 1, 2, 3, 4
    THEN: the position is held from the most recent past event,
          and is zero before the first event
    """
    times = [1.0, 3.0]
    positions = [5.0, 9.0]
    grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    result = simulation.sample_on_grid(times, positions, grid)
    assert np.allclose(result, [0.0, 5.0, 5.0, 9.0, 9.0])


def test_sample_on_grid_starts_at_origin_before_first_event():
    """Before any jump the particle sits at the origin.

    GIVEN: a single event at t=10
    WHEN: the trajectory is sampled only at earlier times
    THEN: every sampled position is zero
    """
    result = simulation.sample_on_grid([10.0], [4.0], [0.0, 1.0, 2.0])
    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_simulate_trajectory_matches_grid_length():
    """A simulated trajectory reports one position per observation time.

    GIVEN: a seeded generator and a time grid
    WHEN: a single trajectory is simulated
    THEN: the result has the same length as the grid
    """
    grid = np.linspace(0.0, 50.0, 25)
    result = simulation.simulate_trajectory(make_generator(), grid)
    assert len(result) == len(grid)


def test_simulate_trajectory_is_reproducible():
    """The same seed reproduces the same trajectory.

    GIVEN: two identically seeded generators and one grid
    WHEN: a trajectory is simulated from each
    THEN: the two trajectories are equal
    """
    grid = np.linspace(0.0, 50.0, 25)
    first = simulation.simulate_trajectory(make_generator(1), grid)
    second = simulation.simulate_trajectory(make_generator(1), grid)
    assert np.allclose(first, second)


def test_ensemble_has_expected_shape():
    """The ensemble is a particles-by-time array.

    GIVEN: a seeded generator, a particle count and a grid
    WHEN: an ensemble is simulated
    THEN: the result has shape (n_particles, len(grid))
    """
    grid = np.linspace(0.0, 20.0, 10)
    result = simulation.simulate_ensemble(make_generator(), 7, grid)
    assert result.shape == (7, 10)


def test_ensemble_particles_start_at_origin():
    """Every particle starts at the origin at the first grid time.

    GIVEN: a grid whose first observation time is zero
    WHEN: an ensemble is simulated
    THEN: the first column is all zeros
    """
    grid = np.linspace(0.0, 20.0, 10)
    result = simulation.simulate_ensemble(make_generator(), 5, grid)
    assert np.allclose(result[:, 0], 0.0)


def test_ensemble_rejects_non_positive_particle_count():
    """A non-positive particle count is rejected.

    GIVEN: a seeded generator and a grid
    WHEN: an ensemble of zero particles is requested
    THEN: a ValueError is raised
    """
    grid = np.linspace(0.0, 20.0, 10)
    with pytest.raises(ValueError):
        simulation.simulate_ensemble(make_generator(), 0, grid)
