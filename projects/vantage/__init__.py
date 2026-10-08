"""Vantage: causal identification and generative scenario analysis.

Contributions of this project:

* the Dutch merit-order effect identified with weather instruments, estimated three ways
  (OLS, IV-2SLS with Newey-West standard errors, and cross-fitted double machine
  learning) so that the endogeneity bias is measured rather than assumed away;
* unsupervised discovery of price regimes with a Gaussian HMM written from first
  principles, including expected regime durations;
* a conditional VAE over daily price profiles, validated against held-out days on the
  distributional statistics that downstream valuation actually depends on;
* a counterfactual build-out simulator built on a learned residual supply curve, which
  reports its own extrapolation share and cross-checks itself against the causal estimate.
"""
