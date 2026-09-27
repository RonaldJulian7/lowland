"""Quantile forecasts -> temporally coherent price scenarios, via a Gaussian copula.

Bellwether gives a marginal distribution per hour. Sampling each hour independently
reproduces those marginals perfectly and the *paths* not at all -- the series ping-pongs
between the 5th and 95th percentile hour to hour, which never happens. Feed that to a
battery optimiser and it sees enormous fake spreads and over-cycles.

What storage cares about is the joint distribution of the path, specifically the spread
between the cheapest and dearest hours in a day. That's dependence structure, not
marginals. A copula separates the two: keep the marginals exactly by inverting each
hour's quantile function, and impose dependence through a correlated Gaussian.

The correlation is parameterised rather than estimated freely -- a full HxH matrix from a
few hundred forecast origins is badly conditioned. Exponential decay in lag, plus a 24 h
periodic term because a forecast that runs high at Tuesday's evening peak tends to run
high at Wednesday's too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from lowland.utils import get_logger

log = get_logger(__name__)


def quantile_function(
    q_row: np.ndarray, quantiles: tuple[float, ...], u: np.ndarray
) -> np.ndarray:
    """Invert an estimated quantile function at probability levels ``u``.

    Between knots this is linear interpolation. Outside the outermost knots it switches
    to exponential tails matched to the outer knot spacing, so a scenario drawn at u =
    0.995 lands somewhere plausible instead of being clipped onto the 95th percentile --
    which would systematically understate spike risk, the single thing a battery is paid
    to capture.
    """
    taus = np.asarray(quantiles, dtype=float)
    out = np.interp(u, taus, q_row)

    lo = u < taus[0]
    if lo.any():
        scale = max(q_row[1] - q_row[0], 1e-6)
        out[lo] = q_row[0] + scale * np.log(u[lo] / taus[0])

    hi = u > taus[-1]
    if hi.any():
        scale = max(q_row[-1] - q_row[-2], 1e-6)
        out[hi] = q_row[-1] - scale * np.log((1.0 - u[hi]) / (1.0 - taus[-1]))

    return out


def parametric_correlation(
    horizon: int, rho: float = 0.82, seasonal_weight: float = 0.25, period: int = 24
) -> np.ndarray:
    """Correlation matrix combining exponential lag decay with a daily periodic term.

    ``rho ** lag`` captures short-range error persistence; the cosine term reinstates the
    same-hour-next-day correlation that pure decay destroys. The result is projected onto
    the nearest positive-definite matrix, because the sum of two valid kernels with an
    arbitrary weight is not automatically one.
    """
    lags = np.abs(np.subtract.outer(np.arange(horizon), np.arange(horizon)))
    decay = rho ** lags
    seasonal = np.cos(2 * np.pi * lags / period)
    corr = (1 - seasonal_weight) * decay + seasonal_weight * np.clip(seasonal, 0, None) * decay ** 0.25
    np.fill_diagonal(corr, 1.0)
    return _nearest_psd(corr)


def _nearest_psd(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the positive semi-definite cone."""
    a = (a + a.T) / 2
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    out = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def estimate_error_correlation(
    predictions: pd.DataFrame, horizon: int, max_lag: int | None = None
) -> tuple[float, float]:
    """Estimate ``(rho, seasonal_weight)`` from realised forecast errors.

    ``predictions`` must have columns ``timestamp``, ``y_true`` and a median column. The
    autocorrelation of the error series at lag 1 gives ``rho``; the ratio of the lag-24
    autocorrelation to what pure decay would predict gives the seasonal weight.
    """
    med_col = next((c for c in predictions.columns if c.startswith("q0.5")), None)
    if med_col is None or "y_true" not in predictions.columns:
        return 0.82, 0.25

    df = predictions.sort_values("timestamp")
    err = (df["y_true"] - df[med_col]).to_numpy(dtype=float)
    err = err[np.isfinite(err)]
    if err.size < 200:
        return 0.82, 0.25

    def acf(x: np.ndarray, lag: int) -> float:
        if lag >= x.size:
            return 0.0
        a, b = x[lag:], x[:-lag]
        s = a.std() * b.std()
        return float(np.mean((a - a.mean()) * (b - b.mean())) / s) if s > 0 else 0.0

    rho = float(np.clip(acf(err, 1), 0.3, 0.97))
    a24 = acf(err, 24)
    implied = rho**24
    seasonal = float(np.clip((a24 - implied) / max(1 - implied, 1e-6), 0.0, 0.6))
    log.info("error correlation estimated: rho=%.3f seasonal_weight=%.3f", rho, seasonal)
    return rho, seasonal


@dataclass
class ScenarioGenerator:
    """Draws price paths from per-hour quantile forecasts via a Gaussian copula."""

    quantiles: tuple[float, ...]
    rho: float = 0.82
    seasonal_weight: float = 0.25
    seed: int = 20260924
    antithetic: bool = True

    def sample(self, q_matrix: np.ndarray, n_scenarios: int = 12) -> np.ndarray:
        """Return ``(n_scenarios, H)`` price paths.

        ``antithetic`` pairs each draw with its mirror image about the median. This is
        variance reduction, and it matters here: with only a dozen scenarios an unlucky
        draw can bias the optimiser's decision, and antithetic pairing guarantees the
        ensemble mean sits where it should.
        """
        H = q_matrix.shape[0]
        rng = np.random.default_rng(self.seed)
        corr = parametric_correlation(H, self.rho, self.seasonal_weight)
        chol = np.linalg.cholesky(corr)

        n_draw = (n_scenarios + 1) // 2 if self.antithetic else n_scenarios
        z = rng.standard_normal((n_draw, H)) @ chol.T
        if self.antithetic:
            z = np.vstack([z, -z])[:n_scenarios]

        u = np.clip(stats.norm.cdf(z), 1e-4, 1 - 1e-4)

        out = np.empty((n_scenarios, H))
        for t in range(H):
            row = q_matrix[t]
            if not np.isfinite(row).all():
                med = np.nanmedian(row)
                out[:, t] = med if np.isfinite(med) else 0.0
                continue
            out[:, t] = quantile_function(row, self.quantiles, u[:, t])
        return out

    def diagnostics(self, q_matrix: np.ndarray, n_scenarios: int = 200) -> pd.DataFrame:
        """Check that sampled scenarios reproduce the intended marginals.

        A copula sampler is easy to get subtly wrong. Comparing the empirical quantiles of
        a large sample against the inputs is a cheap, decisive test.
        """
        s = self.sample(q_matrix, n_scenarios)
        rows = []
        for i, tau in enumerate(self.quantiles):
            target = q_matrix[:, i]
            empirical = np.quantile(s, tau, axis=0)
            ok = np.isfinite(target) & np.isfinite(empirical)
            rows.append(
                {
                    "quantile": tau,
                    "mean_target": float(np.mean(target[ok])),
                    "mean_empirical": float(np.mean(empirical[ok])),
                    "mean_abs_error": float(np.mean(np.abs(target[ok] - empirical[ok]))),
                }
            )
        df = pd.DataFrame(rows)
        # Spread realism: the within-day max-min range is what a battery monetises.
        df.attrs["mean_daily_spread"] = float(np.mean(s.max(axis=1) - s.min(axis=1)))
        return df


def make_scenario_fn(
    q_frame: pd.DataFrame,
    quantiles: tuple[float, ...],
    *,
    n_scenarios: int = 12,
    rho: float = 0.82,
    seasonal_weight: float = 0.25,
):
    """Build the ``scenario_fn(start, end)`` callback expected by :func:`stochastic_mpc`.

    ``q_frame`` is a time-indexed frame of quantile columns covering the whole evaluation
    period; the callback slices out the relevant window and samples paths for it.
    """
    gen = ScenarioGenerator(quantiles=quantiles, rho=rho, seasonal_weight=seasonal_weight)
    arr = q_frame.to_numpy(dtype=float)

    def scenario_fn(start: int, end: int) -> np.ndarray:
        return gen.sample(arr[start:end], n_scenarios=n_scenarios)

    return scenario_fn
