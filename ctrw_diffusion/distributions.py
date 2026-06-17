"""Random draws of waiting times and jump lengths for a continuous-time random walk.

This module is the single place in the package where pseudo-random numbers are
generated. Every public function takes an explicit
:class:`numpy.random.Generator` so that simulations are reproducible and so that
tests can inject a seeded generator and assert on deterministic output.

Two families of distributions are provided for each quantity:

- a *thin-tailed* reference (exponential waiting times, Gaussian jumps) that
  produces ordinary diffusion;
- a *heavy-tailed* power-law (Pareto) option that produces anomalous diffusion.
"""

import numpy as np


def _validate_size(size):
    """Return ``size`` as an int, raising ValueError if it is not positive."""
    if size <= 0:
        raise ValueError("size must be a positive integer, got {!r}".format(size))
    return int(size)


def _validate_scale(scale):
    """Raise ValueError if the scale is not strictly positive."""
    if scale <= 0:
        raise ValueError("scale must be strictly positive, got {!r}".format(scale))


def draw_waiting_times(generator, size, kind="exponential", scale=1.0, exponent=1.5):
    """Draw a sample of positive waiting times. See module docstring for details."""
    size = _validate_size(size)
    _validate_scale(scale)
    if kind == "exponential":
        return generator.exponential(scale=scale, size=size)
    if kind == "pareto":
        _validate_scale(exponent)
        # (pareto + 1) * scale shifts the support to [scale, inf), heavy-tailed.
        return (generator.pareto(a=exponent, size=size) + 1.0) * scale
    raise ValueError(
        "unknown waiting-time kind {!r}; expected 'exponential' or 'pareto'".format(kind)
    )


def draw_jumps(generator, size, kind="gaussian", scale=1.0, exponent=1.5):
    """Draw a sample of jump displacements. See module docstring for details."""
    size = _validate_size(size)
    _validate_scale(scale)
    if kind == "gaussian":
        return generator.normal(loc=0.0, scale=scale, size=size)
    if kind == "pareto":
        _validate_scale(exponent)
        magnitudes = (generator.pareto(a=exponent, size=size) + 1.0) * scale
        signs = generator.choice([-1.0, 1.0], size=size)
        return magnitudes * signs
    raise ValueError(
        "unknown jump kind {!r}; expected 'gaussian' or 'pareto'".format(kind)
    )
