"""Assembly of one-dimensional continuous-time random walk trajectories.

A continuous-time random walk alternates between *waiting* for a random time and
then performing an instantaneous *jump*. A single particle is therefore fully
described by the sequence of times at which jumps occur and the position reached
after each jump.

To compare and average many particles they must be observed at the same instants.
This module therefore provides, in order:

- :func:`event_times`, the clock times of successive jumps;
- :func:`positions_at_events`, the position immediately after each jump;
- :func:`sample_on_grid`, the position of a particle at arbitrary observation
  times, obtained by holding the last position reached before each time;
- :func:`simulate_trajectory`, a single particle sampled on a time grid;
- :func:`simulate_ensemble`, many independent particles on a shared time grid.

All randomness is delegated to :mod:`ctrw_diffusion.distributions`; the functions
in this module that do not draw random numbers are deterministic and are tested
against hand-computed examples.
"""

import numpy as np
from . import distributions


def event_times(waiting_times):
    """Return the clock times at which successive jumps occur."""
    return np.cumsum(np.asarray(waiting_times, dtype=float))


def positions_at_events(jumps):
    """Return the particle position immediately after each jump."""
    return np.cumsum(np.asarray(jumps, dtype=float))


def sample_on_grid(times, positions, time_grid):
    """Sample a step-like trajectory onto a regular grid of observation times."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    # For each grid time, count how many events have already happened. That count
    # is the index (plus one) into positions; zero means "still at the origin".
    indices = np.searchsorted(times, time_grid, side="right")
    sampled = np.zeros(len(time_grid), dtype=float)
    moved = indices > 0
    sampled[moved] = positions[indices[moved] - 1]
    return sampled


def _draw_until_covered(generator, horizon, waiting_kwargs, jump_kwargs):
    """Draw waiting times and jumps until the event times cover ``horizon``."""
    block = 64
    waiting_parts = []
    total = 0.0
    while total < horizon:
        part = distributions.draw_waiting_times(generator, block, **waiting_kwargs)
        waiting_parts.append(part)
        total += float(np.sum(part))
    waiting = np.concatenate(waiting_parts)
    jumps = distributions.draw_jumps(generator, len(waiting), **jump_kwargs)
    return waiting, jumps


def simulate_trajectory(generator, time_grid, waiting_kwargs=None, jump_kwargs=None):
    """Simulate a single particle and sample it on a time grid."""
    waiting_kwargs = dict(waiting_kwargs or {})
    jump_kwargs = dict(jump_kwargs or {})
    time_grid = np.asarray(time_grid, dtype=float)
    horizon = float(time_grid[-1]) if len(time_grid) else 0.0
    waiting, jumps = _draw_until_covered(
        generator, horizon, waiting_kwargs, jump_kwargs
    )
    times = event_times(waiting)
    positions = positions_at_events(jumps)
    return sample_on_grid(times, positions, time_grid)


def simulate_ensemble(generator, n_particles, time_grid, waiting_kwargs=None, jump_kwargs=None):
    """Simulate many independent particles on a shared time grid."""
    if n_particles <= 0:
        raise ValueError(
            "n_particles must be positive, got {!r}".format(n_particles)
        )
    time_grid = np.asarray(time_grid, dtype=float)
    rows = [
        simulate_trajectory(generator, time_grid, waiting_kwargs, jump_kwargs)
        for _ in range(int(n_particles))
    ]
    return np.vstack(rows)
