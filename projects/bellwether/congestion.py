"""Predictive distribution -> operational congestion risk.

"12,400 MW" isn't a decision. An operator needs how likely the threshold is to break, by
how much, and when in the horizon. All three come out of the full distribution, which is
the practical payoff of forecasting quantiles at all.

We only estimate seven quantile levels, so everything here rests on rebuilding a usable
CDF from those knots: linear interpolation between them, exponential tails beyond the
outermost two. Linear extrapolation in the tails would put mass at physically impossible
values and make the tail probability depend on an arbitrary clipping choice. Far-tail
exceedance numbers are extrapolations; they're labelled as such where they're reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lowland.utils import get_logger

log = get_logger(__name__)


def cdf_at(
    q_matrix: np.ndarray, quantiles: tuple[float, ...], threshold: float | np.ndarray
) -> np.ndarray:
    """Estimate ``P(Y <= threshold)`` for each row of ``q_matrix``.

    Parameters
    ----------
    q_matrix:
        ``(n, n_quantiles)`` predicted quantiles, ascending along axis 1.
    quantiles:
        The levels corresponding to the columns.
    threshold:
        Scalar, or one value per row.
    """
    taus = np.asarray(quantiles, dtype=float)
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), (q_matrix.shape[0],))
    out = np.empty(q_matrix.shape[0], dtype=float)

    for i in range(q_matrix.shape[0]):
        qs = q_matrix[i]
        t = thr[i]
        if not np.isfinite(qs).all() or not np.isfinite(t):
            out[i] = np.nan
            continue

        if t <= qs[0]:
            # Lower tail: exponential decay fitted to the two lowest knots.
            scale = max(qs[1] - qs[0], 1e-6)
            out[i] = taus[0] * np.exp((t - qs[0]) / scale)
        elif t >= qs[-1]:
            # Upper tail: exceedance decays exponentially above the top knot.
            scale = max(qs[-1] - qs[-2], 1e-6)
            out[i] = 1.0 - (1.0 - taus[-1]) * np.exp(-(t - qs[-1]) / scale)
        else:
            out[i] = float(np.interp(t, qs, taus))

    return np.clip(out, 0.0, 1.0)


def exceedance_probability(
    q_matrix: np.ndarray, quantiles: tuple[float, ...], threshold: float | np.ndarray
) -> np.ndarray:
    """``P(Y > threshold)``: the headline congestion-risk number."""
    return 1.0 - cdf_at(q_matrix, quantiles, threshold)


def expected_exceedance(
    q_matrix: np.ndarray,
    quantiles: tuple[float, ...],
    threshold: float,
    *,
    n_grid: int = 200,
) -> np.ndarray:
    """``E[max(Y - threshold, 0)]`` in MW, by integrating the survival function.

    This is the quantity that scales with cost: a 5% chance of a 3,000 MW breach and a
    40% chance of a 200 MW breach are very different operational problems, and an
    exceedance *probability* alone cannot tell them apart.
    """
    taus = np.asarray(quantiles, dtype=float)
    out = np.zeros(q_matrix.shape[0])
    for i in range(q_matrix.shape[0]):
        qs = q_matrix[i]
        if not np.isfinite(qs).all():
            out[i] = np.nan
            continue
        hi = max(qs[-1] + 3.0 * max(qs[-1] - qs[-2], 1.0), threshold + 1.0)
        grid = np.linspace(threshold, hi, n_grid)
        surv = 1.0 - cdf_at(np.tile(qs, (n_grid, 1)), tuple(taus), grid)
        out[i] = float(np.trapezoid(surv, grid))
    return out


def value_at_risk(q_matrix: np.ndarray, quantiles: tuple[float, ...], level: float = 0.95) -> np.ndarray:
    """The ``level`` quantile of the predictive distribution, interpolated across knots."""
    taus = np.asarray(quantiles, dtype=float)
    return np.array([np.interp(level, taus, row) for row in q_matrix])


def conditional_value_at_risk(
    q_matrix: np.ndarray, quantiles: tuple[float, ...], level: float = 0.95, n_grid: int = 100
) -> np.ndarray:
    """Expected value conditional on exceeding the ``level`` quantile (expected shortfall).

    CVaR is coherent where VaR is not: it is sensitive to how bad the tail actually is,
    not merely to where it starts.
    """
    taus = np.asarray(quantiles, dtype=float)
    grid = np.linspace(level, 0.999, n_grid)
    out = np.empty(q_matrix.shape[0])
    for i, row in enumerate(q_matrix):
        if not np.isfinite(row).all():
            out[i] = np.nan
            continue
        vals = np.interp(grid, taus, row)
        # Beyond the top knot, extend with the fitted exponential tail.
        top = grid > taus[-1]
        if top.any():
            scale = max(row[-1] - row[-2], 1e-6)
            vals[top] = row[-1] - scale * np.log((1.0 - grid[top]) / (1.0 - taus[-1]))
        out[i] = float(np.mean(vals))
    return out


@dataclass
class CongestionReport:
    """Per-hour risk table plus horizon-level summaries."""

    table: pd.DataFrame
    threshold_mw: float
    peak_risk_hour: pd.Timestamp | None
    hours_above_50pct: int
    max_expected_exceedance_mw: float
    total_expected_energy_mwh: float

    def summary(self) -> dict[str, object]:
        return {
            "threshold_mw": round(self.threshold_mw, 1),
            "peak_risk_hour": str(self.peak_risk_hour) if self.peak_risk_hour is not None else None,
            "hours_above_50pct": self.hours_above_50pct,
            "max_expected_exceedance_mw": round(self.max_expected_exceedance_mw, 1),
            "total_expected_energy_mwh": round(self.total_expected_energy_mwh, 1),
        }


def build_congestion_report(
    index: pd.DatetimeIndex,
    q_matrix: np.ndarray,
    quantiles: tuple[float, ...],
    threshold_mw: float,
    *,
    ramp_threshold_mw: float | None = None,
) -> CongestionReport:
    """Assemble the full risk table for one forecast horizon.

    Includes a *ramp* risk column when ``ramp_threshold_mw`` is given. Ramp rate is often
    the binding constraint rather than level: the Dutch system can serve 14 GW, but not if
    it has to get there from 9 GW in two hours.
    """
    p_exceed = exceedance_probability(q_matrix, quantiles, threshold_mw)
    exp_exceed = expected_exceedance(q_matrix, quantiles, threshold_mw)
    var95 = value_at_risk(q_matrix, quantiles, 0.95)
    cvar95 = conditional_value_at_risk(q_matrix, quantiles, 0.95)

    med_i = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    median = q_matrix[:, med_i]

    table = pd.DataFrame(
        {
            "median_mw": median,
            "p_exceed": p_exceed,
            "expected_exceedance_mw": exp_exceed,
            "var95_mw": var95,
            "cvar95_mw": cvar95,
            "interval_width_mw": q_matrix[:, -1] - q_matrix[:, 0],
        },
        index=index,
    )

    if ramp_threshold_mw is not None:
        # Successive-hour uncertainty is correlated, so differencing the medians
        # understates ramp risk. A conservative bound uses the upper quantile of the
        # later hour against the lower quantile of the earlier one.
        up_ramp = np.r_[np.nan, q_matrix[1:, -2] - q_matrix[:-1, 1]]
        table["worst_case_ramp_mw"] = up_ramp
        table["ramp_flag"] = (up_ramp > ramp_threshold_mw).astype(float)

    peak_hour = table["p_exceed"].idxmax() if table["p_exceed"].notna().any() else None
    return CongestionReport(
        table=table,
        threshold_mw=threshold_mw,
        peak_risk_hour=peak_hour,
        hours_above_50pct=int((table["p_exceed"] > 0.5).sum()),
        max_expected_exceedance_mw=float(np.nanmax(exp_exceed)) if len(exp_exceed) else 0.0,
        total_expected_energy_mwh=float(np.nansum(exp_exceed)),
    )


def risk_calibration(
    y_true: np.ndarray,
    q_matrix: np.ndarray,
    quantiles: tuple[float, ...],
    threshold: float,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability of the exceedance probabilities themselves.

    Bins forecasts by predicted ``P(Y > threshold)`` and compares each bin's mean
    prediction to the realised exceedance frequency. This is the check that decides
    whether the risk number can be trusted as a probability -- a model can have excellent
    CRPS and still produce badly calibrated threshold probabilities, because CRPS averages
    over the whole distribution while the operator only cares about one point of it.
    """
    p = exceedance_probability(q_matrix, quantiles, threshold)
    actual = (y_true > threshold).astype(float)
    ok = np.isfinite(p) & np.isfinite(actual)
    p, actual = p[ok], actual[ok]
    if p.size == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "observed_freq"])

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}",
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_freq": float(actual[m].mean()),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        # Brier score decomposes into reliability + resolution - uncertainty; the overall
        # value is reported alongside the bins so a single number can be quoted.
        df.attrs["brier"] = float(np.mean((p - actual) ** 2))
        base = actual.mean()
        df.attrs["brier_climatology"] = float(np.mean((base - actual) ** 2))
        df.attrs["brier_skill"] = float(1 - df.attrs["brier"] / max(df.attrs["brier_climatology"], 1e-9))
    return df
