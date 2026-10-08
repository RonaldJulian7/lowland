"""Bellwether: probabilistic forecasting of Dutch residual load and day-ahead price.

Contributions of this project:

* a like-for-like comparison of gradient-boosted quantile regression against a
  sequence-to-sequence quantile network on the same folds and the same features;
* conformal calibration (split, Mondrian group-conditional, and online adaptive) giving
  distribution-free coverage guarantees that survive the regime shifts in this data;
* translation of the predictive distribution into operational congestion risk --
  exceedance probability, expected exceedance energy and ramp risk -- with the
  calibration of those risk numbers itself verified.
"""
