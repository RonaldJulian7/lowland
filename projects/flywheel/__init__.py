"""Flywheel: battery dispatch under forecast uncertainty.

Contributions of this project:

* a perfect-foresight MILP that bounds what any controller could earn, so every result is
  reported as a fraction of the achievable optimum rather than as a bare euro figure;
* a two-stage stochastic MILP with non-anticipativity and an optional CVaR objective,
  fed by temporally coherent price scenarios drawn through a Gaussian copula from project
  1's predictive quantiles;
* a PPO agent learned directly from market interaction, benchmarked against both the
  optimiser and a no-forecast heuristic;
* a profit-versus-congestion Pareto frontier quantifying what grid-friendly operation
  costs the asset owner.
"""
