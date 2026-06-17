"""ctrw_diffusion: a continuous-time random walk simulator for anomalous diffusion.

The package is organised in three independent stages that mirror a reproducible
simulation pipeline:

- ``distributions``: random draws of waiting times and jump lengths (the only
  place where randomness is used, so it can be seeded and isolated in tests);
- ``simulation``: assembly of particle trajectories from those draws;
- ``analysis``: deterministic computation of the mean-squared displacement and
  estimation of the anomalous-diffusion exponent.

Visualization and input/output helpers are provided separately so that the
scientific logic stays free of plotting and file-system concerns.
"""

__version__ = (0, 1, 0)
