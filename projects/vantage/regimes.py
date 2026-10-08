"""Price regimes via a Gaussian HMM, fitted with Baum-Welch.

Dutch prices aren't drawn from one distribution. The mean went from ~EUR 32/MWh to
~EUR 242/MWh and back between 2015 and 2026, and the shape changed too -- negative hours
went from a curiosity to a routine feature of windy Sunday afternoons. Any unconditional
"price distribution" is mis-specified, and everything built on it inherits that.

An HMM gives two things nothing else does: expected regime duration (off the transition
matrix diagonal), which tells you whether today's spread is likely to hold; and a
per-day probability of being in each state, usable as a feature or a risk flag.

Written out rather than pulled from hmmlearn. The forward-backward recursions are where
the numerical care lives -- all of it in log space, because underflow on long sequences
corrupts things silently rather than loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from lowland.utils import get_logger

log = get_logger(__name__)


@dataclass
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM fitted by Baum-Welch in log space.

    Parameters
    ----------
    n_states:
        Number of latent regimes.
    n_iter, tol:
        EM iteration cap and relative log-likelihood convergence tolerance.
    min_var:
        Variance floor. Without it, EM can collapse a state onto a single observation and
        drive its variance to zero, which sends the log-likelihood to infinity and the
        parameters to nonsense -- the classic degenerate solution of Gaussian mixtures.
    """

    n_states: int = 3
    n_iter: int = 120
    tol: float = 1e-4
    min_var: float = 1e-3
    seed: int = 20260924

    start_: np.ndarray | None = None
    trans_: np.ndarray | None = None
    means_: np.ndarray | None = None
    vars_: np.ndarray | None = None
    loglik_: list[float] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)

    # ---- emissions -------------------------------------------------------------------

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """``(n_samples, n_states)`` log density of each observation under each state."""
        n, d = X.shape
        out = np.empty((n, self.n_states))
        for k in range(self.n_states):
            var = self.vars_[k]
            diff = X - self.means_[k]
            out[:, k] = -0.5 * (
                d * np.log(2 * np.pi) + np.log(var).sum() + ((diff**2) / var).sum(axis=1)
            )
        return out

    # ---- forward-backward -------------------------------------------------------------

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        n = log_b.shape[0]
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = np.log(self.start_ + 1e-300) + log_b[0]
        log_trans = np.log(self.trans_ + 1e-300)
        for t in range(1, n):
            log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        n = log_b.shape[0]
        log_beta = np.zeros_like(log_b)
        log_trans = np.log(self.trans_ + 1e-300)
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(log_trans + log_b[t + 1] + log_beta[t + 1], axis=1)
        return log_beta

    # ---- fitting -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> GaussianHMM:
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.feature_names_ = feature_names or [f"f{i}" for i in range(d)]

        # Initialise the state means on quantiles of the first feature rather than at
        # random: price regimes are ordered by level, so this starts EM in the right basin
        # and makes the fit reproducible instead of seed-dependent.
        qs = np.linspace(0.15, 0.85, self.n_states)
        self.means_ = np.array([np.quantile(X, q, axis=0) for q in qs])
        self.means_ += rng.normal(0, 1e-3, self.means_.shape)
        self.vars_ = np.tile(X.var(axis=0) + self.min_var, (self.n_states, 1))
        self.start_ = np.full(self.n_states, 1.0 / self.n_states)
        # Initialise transitions as strongly persistent: regimes last weeks, not hours.
        self.trans_ = np.full((self.n_states, self.n_states), 0.01 / max(self.n_states - 1, 1))
        np.fill_diagonal(self.trans_, 0.99)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_b = self._log_emission(X)
            log_alpha, ll = self._forward(log_b)
            log_beta = self._backward(log_b)

            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_trans = np.log(self.trans_ + 1e-300)
            xi_sum = np.zeros((self.n_states, self.n_states))
            for t in range(n - 1):
                m = (
                    log_alpha[t][:, None]
                    + log_trans
                    + log_b[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                )
                xi_sum += np.exp(m - logsumexp(m))

            self.start_ = gamma[0] / gamma[0].sum()
            self.trans_ = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)

            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / np.maximum(w[:, None], 1e-300)
            for k in range(self.n_states):
                diff = X - self.means_[k]
                self.vars_[k] = np.maximum(
                    (gamma[:, k][:, None] * diff**2).sum(axis=0) / max(w[k], 1e-300), self.min_var
                )

            self.loglik_.append(ll)
            if it > 2 and abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                log.info("HMM converged after %s iterations (loglik=%.1f)", it, ll)
                break
            prev_ll = ll
        else:
            log.info("HMM hit the iteration cap (loglik=%.1f)", prev_ll)

        self._reorder_by_level()
        return self

    def _reorder_by_level(self) -> None:
        """Sort states by the mean of the first feature so labels are stable across runs."""
        order = np.argsort(self.means_[:, 0])
        self.means_ = self.means_[order]
        self.vars_ = self.vars_[order]
        self.start_ = self.start_[order]
        self.trans_ = self.trans_[np.ix_(order, order)]

    # ---- inference ----------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed posterior probability of each state at each time."""
        log_b = self._log_emission(np.asarray(X, dtype=float))
        log_alpha, _ = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely state sequence by Viterbi decoding.

        Viterbi rather than per-timestep argmax: the latter can return a path that the
        transition matrix assigns zero probability, which is incoherent when the point of
        the model is the dynamics.
        """
        X = np.asarray(X, dtype=float)
        log_b = self._log_emission(X)
        n = len(X)
        log_trans = np.log(self.trans_ + 1e-300)

        delta = np.empty((n, self.n_states))
        psi = np.zeros((n, self.n_states), dtype=int)
        delta[0] = np.log(self.start_ + 1e-300) + log_b[0]
        for t in range(1, n):
            m = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(m, axis=0)
            delta[t] = np.max(m, axis=0) + log_b[t]

        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # ---- reporting -----------------------------------------------------------------------

    def n_parameters(self) -> int:
        k, d = self.n_states, self.means_.shape[1]
        return k - 1 + k * (k - 1) + k * d * 2

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion, for choosing the number of states."""
        _, ll = self._forward(self._log_emission(np.asarray(X, dtype=float)))
        return float(-2 * ll + self.n_parameters() * np.log(len(X)))

    def summary(self) -> pd.DataFrame:
        """Per-state means, dispersion and expected dwell time."""
        rows = []
        for k in range(self.n_states):
            p_stay = float(self.trans_[k, k])
            rows.append(
                {
                    "state": k,
                    **{f"mean_{nm}": float(self.means_[k, i])
                       for i, nm in enumerate(self.feature_names_)},
                    **{f"sd_{nm}": float(np.sqrt(self.vars_[k, i]))
                       for i, nm in enumerate(self.feature_names_)},
                    "p_stay": round(p_stay, 4),
                    # Expected dwell time of a geometric distribution with success 1 - p.
                    "expected_duration_h": round(1.0 / max(1 - p_stay, 1e-9), 1),
                }
            )
        return pd.DataFrame(rows)


def fit_price_regimes(
    panel: pd.DataFrame,
    n_states: int = 3,
    *,
    resample: str = "D",
    select_states: tuple[int, ...] | None = None,
) -> tuple[GaussianHMM, pd.DataFrame, pd.DataFrame]:
    """Fit regimes on daily price level, volatility and spread.

    Daily rather than hourly aggregation is deliberate: at hourly resolution the model
    spends its capacity describing the within-day shape, which is already well explained
    by the diurnal cycle, instead of the slow-moving market conditions the regimes are
    meant to capture.

    ``select_states`` fits several state counts and reports BIC for each, so the choice is
    made by a criterion rather than by eye.
    """
    price = panel.loc[panel.get("is_provisional", 0) == 0, "price_da"].dropna()
    daily = pd.DataFrame(
        {
            "level": price.resample(resample).mean(),
            "volatility": price.resample(resample).std(),
            "spread": price.resample(resample).max() - price.resample(resample).min(),
            "neg_hours": (price < 0).resample(resample).sum(),
        }
    ).dropna()

    # Standardise so no single feature dominates the diagonal-covariance likelihood
    # purely because of its units.
    mu, sd = daily.mean(), daily.std().replace(0, 1.0)
    Z = ((daily - mu) / sd).to_numpy()

    bic_rows = []
    if select_states:
        for k in select_states:
            m = GaussianHMM(n_states=k).fit(Z, list(daily.columns))
            bic_rows.append({"n_states": k, "bic": m.bic(Z), "loglik": m.loglik_[-1]})
        bic_df = pd.DataFrame(bic_rows)
        n_states = int(bic_df.loc[bic_df["bic"].idxmin(), "n_states"])
        log.info("BIC selects %s states", n_states)
    else:
        bic_df = pd.DataFrame()

    model = GaussianHMM(n_states=n_states).fit(Z, list(daily.columns))
    states = model.predict(Z)
    proba = model.predict_proba(Z)

    assign = daily.copy()
    assign["state"] = states
    for k in range(model.n_states):
        assign[f"p_state{k}"] = proba[:, k]

    # Report the state means back in original units, which is what a reader can interpret.
    summary = model.summary()
    for i, col in enumerate(daily.columns):
        summary[f"mean_{col}"] = summary[f"mean_{col}"] * sd[col] + mu[col]
        summary[f"sd_{col}"] = summary[f"sd_{col}"] * sd[col]
    summary["n_days"] = [int((states == k).sum()) for k in range(model.n_states)]
    summary["share"] = (summary["n_days"] / len(states)).round(3)

    return model, assign, (bic_df if len(bic_df) else summary)
