"""lowland - shared library for the Lowland portfolio.

Three projects sit on top of this package:

``projects.bellwether``
    Probabilistic forecasting of Dutch residual load and day-ahead price, with
    distribution-free coverage guarantees via conformal prediction.

``projects.flywheel``
    Battery dispatch under forecast uncertainty: a perfect-foresight MILP upper bound,
    deterministic and stochastic model-predictive control, and a PPO agent.

``projects.vantage``
    Causal identification of the merit-order effect and a generative scenario engine for
    counterfactual build-out pathways.
"""

__version__ = "0.1.0"
