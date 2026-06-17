"""Tests for the :mod:`ctrw_diffusion.distributions` module.

Randomness is made deterministic by passing a seeded
:class:`numpy.random.Generator` to every function under test, following the
principle that tests must reliably pass or fail.
"""

import numpy as np
import pytest

from ctrw_diffusion import distributions


def make_generator(seed=12345):
    """Return a freshly seeded generator for deterministic tests."""
    return np.random.default_rng(seed)


def test_waiting_times_have_requested_size():
    """The number of waiting times returned matches the request.

    GIVEN: a seeded generator and a requested sample size
    WHEN: waiting times are drawn
    THEN: the returned array has exactly that many elements
    """
    generator = make_generator()
    result = distributions.draw_waiting_times(generator, size=100)
    assert len(result) == 100


def test_waiting_times_are_strictly_positive():
    """Waiting times are physical durations and must be positive.

    GIVEN: a seeded generator
    WHEN: a sample of waiting times is drawn
    THEN: every value is strictly greater than zero
    """
    generator = make_generator()
    result = distributions.draw_waiting_times(generator, size=1000)
    assert np.all(result > 0)


def test_waiting_times_are_reproducible_with_same_seed():
    """The same seed yields identical waiting times.

    GIVEN: two generators seeded identically
    WHEN: waiting times are drawn from each
    THEN: the two samples are element-wise equal
    """
    first = distributions.draw_waiting_times(make_generator(7), size=50)
    second = distributions.draw_waiting_times(make_generator(7), size=50)
    assert np.array_equal(first, second)


def test_jumps_have_requested_size():
    """The number of jumps returned matches the request.

    GIVEN: a seeded generator and a requested sample size
    WHEN: jumps are drawn
    THEN: the returned array has exactly that many elements
    """
    generator = make_generator()
    result = distributions.draw_jumps(generator, size=100)
    assert len(result) == 100


def test_gaussian_jumps_take_both_signs():
    """Symmetric jumps must be able to move in both directions.

    GIVEN: a seeded generator
    WHEN: a large sample of Gaussian jumps is drawn
    THEN: at least one positive and one negative jump appear
    """
    generator = make_generator()
    result = distributions.draw_jumps(generator, size=1000, kind="gaussian")
    assert np.any(result > 0)
    assert np.any(result < 0)


def test_unknown_waiting_time_kind_raises():
    """An unsupported distribution name is reported clearly.

    GIVEN: a seeded generator and an invalid distribution name
    WHEN: waiting times are requested
    THEN: a ValueError is raised
    """
    generator = make_generator()
    with pytest.raises(ValueError):
        distributions.draw_waiting_times(generator, size=10, kind="not_a_kind")
