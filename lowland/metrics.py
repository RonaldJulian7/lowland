"""Scoring rules and tests for probabilistic forecasts.

The thing being measured is sharpness subject to calibration. An interval that always
covers is useless if it spans the whole plausible range; a tight one that misses half the
time is worse. So: pinball and CRPS are proper (can't be gamed by misreporting), coverage
and the reliability curve isolate calibration, interval width isolates sharpness, and
Diebold-Mariano says whether a difference between two models is real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent.

    Preferred over plain MAPE here because day-ahead prices pass through and below zero,
    where MAPE is undefined or explodes.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray, season: int = 24) -> float:
    """Mean absolute scaled error against an in-sample seasonal-naive benchmark.

    Scale-free and defined for series that cross zero, which makes it the one point
    metric comparable across load (MW) and price (EUR/MWh).
    """
    naive_err = np.abs(np.diff(y_insample, n=1 * season))
    scale = np.mean(naive_err) if naive_err.size else np.nan
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def skill_score(score_model: float, score_ref: float) -> float:
    """Fractional improvement over a reference score; 1.0 is perfect, 0.0 is no better."""
    if not np.isfinite(score_ref) or score_ref == 0:
        return float("nan")
    return float(1.0 - score_model / score_ref)


# --------------------------------------------------------------------------------------
# Probabilistic metrics
# --------------------------------------------------------------------------------------


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    """Quantile (pinball) loss at level ``tau``.

    Minimised in expectation by the true ``tau``-quantile, which is what makes it proper
    for quantile estimation.
    """
    d = y_true - q_pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def mean_pinball(y_true: np.ndarray, q_matrix: np.ndarray, quantiles: tuple[float, ...]) -> float:
    """Pinball loss averaged over all estimated quantile levels."""
    return float(np.mean([pinball_loss(y_true, q_matrix[:, i], t) for i, t in enumerate(quantiles)]))


def crps_from_quantiles(
    y_true: np.ndarray, q_matrix: np.ndarray, quantiles: tuple[float, ...]
) -> float:
    """Approximate CRPS by numerically integrating the pinball loss over ``tau``.

    CRPS equals twice the integral of the pinball loss over the unit interval. With a
    finite quantile grid this is a trapezoidal approximation; it is slightly optimistic
    in the tails beyond the outermost estimated quantile, so it is reported alongside the
    raw pinball loss rather than instead of it.
    """
    taus = np.asarray(quantiles, dtype=float)
    losses = np.array([pinball_loss(y_true, q_matrix[:, i], t) for i, t in enumerate(taus)])
    order = np.argsort(taus)
    return float(2.0 * np.trapezoid(losses[order], taus[order]))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction interval coverage probability (PICP), in ``[0, 1]``."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction interval width (MPIW), in the units of the target."""
    return float(np.mean(upper - lower))


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Winkler / interval score for a central ``(1 - alpha)`` interval.

    Penalises width plus a ``2/alpha`` multiple of any miss, so it is minimised only by an
    interval that is both narrow and correctly calibrated. Lower is better.
    """
    width = upper - lower
    below = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    above = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + below + above))


def reliability_curve(
    y_true: np.ndarray, q_matrix: np.ndarray, quantiles: tuple[float, ...]
) -> pd.DataFrame:
    """Empirical vs nominal exceedance rates, one row per quantile level.

    A perfectly calibrated model puts ``empirical`` on the 45-degree line. Systematic
    deviation above the line means the predictive distribution is too narrow.
    """
    rows = []
    for i, tau in enumerate(quantiles):
        emp = float(np.mean(y_true <= q_matrix[:, i]))
        rows.append({"nominal": tau, "empirical": emp, "gap": emp - tau})
    return pd.DataFrame(rows)


def quantile_crossing_rate(q_matrix: np.ndarray) -> float:
    """Fraction of predictions where estimated quantiles are not monotone in ``tau``.

    Independently fitted quantile models can cross, which makes the implied distribution
    invalid. Reported so the downstream rearrangement step can be justified rather than
    applied silently.
    """
    diffs = np.diff(q_matrix, axis=1)
    return float(np.mean((diffs < 0).any(axis=1)))


def rearrange_quantiles(q_matrix: np.ndarray) -> np.ndarray:
    """Enforce monotonicity by sorting each row.

    This is Chernozhukov, Fernandez-Val and Galichon's rearrangement: sorting a set of
    crossing quantile estimates is guaranteed not to increase estimation error, so it is a
    free fix rather than a cosmetic one.
    """
    return np.sort(q_matrix, axis=1)


# --------------------------------------------------------------------------------------
# Statistical comparison
# --------------------------------------------------------------------------------------


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Returns ``(statistic, two_sided_p_value)``. A negative statistic favours model A.

    Forecast errors at horizon ``h`` are autocorrelated up to lag ``h - 1``, so the
    variance of the mean loss differential uses a Newey-West correction with that
    bandwidth. Ignoring it is the standard way to overstate significance in forecasting
    papers.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return float("nan"), float("nan")

    d_bar = d.mean()
    gamma0 = np.sum((d - d_bar) ** 2) / n
    var = gamma0
    for lag in range(1, h):
        cov = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / n
        var += 2.0 * (1.0 - lag / h) * cov
    if var <= 0:
        return float("nan"), float("nan")

    stat = d_bar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction.
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= corr
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def evaluate_probabilistic(
    y_true: np.ndarray,
    q_matrix: np.ndarray,
    quantiles: tuple[float, ...],
    *,
    y_insample: np.ndarray | None = None,
    label: str = "model",
) -> dict[str, float]:
    """Compute the full metric suite for one probabilistic forecast.

    ``q_matrix`` has shape ``(n_samples, n_quantiles)`` with columns ordered as
    ``quantiles``. The median column is used as the point forecast.
    """
    finite = np.isfinite(y_true) & np.isfinite(q_matrix).all(axis=1)
    y_true = y_true[finite]
    q_matrix = q_matrix[finite]
    if y_true.size == 0:
        return {"label": label, "n": 0}  # type: ignore[dict-item]

    med_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    point = q_matrix[:, med_idx]

    out: dict[str, float] = {
        "n": float(y_true.size),
        "mae": mae(y_true, point),
        "rmse": rmse(y_true, point),
        "smape": smape(y_true, point),
        "pinball": mean_pinball(y_true, q_matrix, quantiles),
        "crps": crps_from_quantiles(y_true, q_matrix, quantiles),
        "crossing_rate": quantile_crossing_rate(q_matrix),
    }
    if y_insample is not None and y_insample.size > 48:
        out["mase"] = mase(y_true, point, y_insample)

    for lo_tau, hi_tau in ((0.05, 0.95), (0.10, 0.90), (0.25, 0.75)):
        if lo_tau in quantiles and hi_tau in quantiles:
            lo = q_matrix[:, quantiles.index(lo_tau)]
            hi = q_matrix[:, quantiles.index(hi_tau)]
            nom = int(round(100 * (hi_tau - lo_tau)))
            out[f"picp{nom}"] = coverage(y_true, lo, hi)
            out[f"mpiw{nom}"] = mean_interval_width(lo, hi)
            out[f"winkler{nom}"] = interval_score(y_true, lo, hi, alpha=1 - (hi_tau - lo_tau))

    return out


def metrics_frame(results: list[dict[str, float]], index_col: str = "label") -> pd.DataFrame:
    """Stack per-model metric dictionaries into a comparison table."""
    df = pd.DataFrame(results)
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df.round(4)
