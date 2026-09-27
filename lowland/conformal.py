"""Conformal prediction: distribution-free coverage on top of any quantile model.

A quantile model that reports a 90% interval almost never delivers 90% out of sample.
It's fitted on the training distribution, and any mismatch with the deployment one shows
up as miscalibration. On this data the mismatch is severe -- the gas crisis rewrote the
price distribution and renewables build-out keeps moving residual load.

CQR (Romano, Patterson & Candes 2019) fixes it with a finite-sample guarantee and no
distributional assumptions: measure how far outside the predicted interval the truth fell
on a held-out calibration set, then widen (or narrow) by the right empirical quantile of
that miss. Gives P(Y in C(X)) >= 1 - alpha for exchangeable data, whatever the model.

Three flavours here: plain split CQR, a Mondrian version that calibrates within groups
(marginal coverage is weak -- you can hit 90% overall and 60% on winter evening peaks,
which are the hours anyone actually cares about), and an online adaptive controller that
drops exchangeability altogether and feeds back on the realised miss rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lowland.utils import get_logger

log = get_logger(__name__)


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """The conformal quantile with the finite-sample correction.

    Uses the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile rather than the plain
    ``1 - alpha`` one. The ``+1`` is what converts an asymptotic statement into an exact
    finite-sample guarantee; dropping it undercovers slightly at small ``n``.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """CQR non-conformity scores.

    ``E_i = max(lower_i - y_i, y_i - upper_i)``: positive when the truth fell outside the
    interval (by how much), negative when it fell inside (how much slack there was). The
    signed form is what lets CQR *tighten* an over-wide interval as well as widen a
    too-narrow one.
    """
    return np.maximum(lower - y, y - upper)


@dataclass
class ConformalQuantileRegressor:
    """Split conformal calibration of a set of predicted quantiles.

    Parameters
    ----------
    quantiles:
        The quantile levels the base model produces, ascending.
    """

    quantiles: tuple[float, ...]
    corrections_: dict[tuple[float, float], float] = field(default_factory=dict)
    fitted_: bool = False

    def _pairs(self) -> list[tuple[int, int, float]]:
        """Symmetric (lower, upper) index pairs and their miscoverage level ``alpha``."""
        q = np.asarray(self.quantiles)
        pairs = []
        n = len(q)
        for i in range(n // 2):
            j = n - 1 - i
            if j <= i:
                break
            alpha = float(q[i] + (1 - q[j]))
            pairs.append((i, j, alpha))
        return pairs

    def fit(self, y_calib: np.ndarray, q_calib: np.ndarray) -> ConformalQuantileRegressor:
        """Learn one correction per symmetric interval from held-out calibration data."""
        y_calib = np.asarray(y_calib, dtype=float)
        ok = np.isfinite(y_calib) & np.isfinite(q_calib).all(axis=1)
        y_calib, q_calib = y_calib[ok], q_calib[ok]

        self.corrections_ = {}
        for i, j, alpha in self._pairs():
            scores = cqr_scores(y_calib, q_calib[:, i], q_calib[:, j])
            self.corrections_[(self.quantiles[i], self.quantiles[j])] = _finite_sample_quantile(
                scores, alpha
            )
        self.fitted_ = True
        log.info(
            "CQR fitted on n=%s; corrections=%s",
            len(y_calib),
            {f"{a:.2f}-{b:.2f}": round(v, 2) for (a, b), v in self.corrections_.items()},
        )
        return self

    def transform(self, q_pred: np.ndarray) -> np.ndarray:
        """Apply the learned corrections, returning calibrated quantiles."""
        if not self.fitted_:
            raise RuntimeError("call fit() before transform()")
        out = q_pred.copy()
        for i, j, _ in self._pairs():
            delta = self.corrections_[(self.quantiles[i], self.quantiles[j])]
            out[:, i] = q_pred[:, i] - delta
            out[:, j] = q_pred[:, j] + delta
        # The median is left untouched: CQR calibrates intervals, not the point forecast.
        return np.sort(out, axis=1)

    def fit_transform(
        self, y_calib: np.ndarray, q_calib: np.ndarray, q_test: np.ndarray
    ) -> np.ndarray:
        return self.fit(y_calib, q_calib).transform(q_test)


@dataclass
class MondrianConformalQuantileRegressor:
    """Group-conditional (Mondrian) CQR.

    Calibration is performed separately within each group, so coverage holds *within* the
    group rather than only on average across groups. Groups with too few calibration
    points fall back to the pooled correction, which keeps the estimator stable.
    """

    quantiles: tuple[float, ...]
    min_group_n: int = 100
    group_models_: dict[object, ConformalQuantileRegressor] = field(default_factory=dict)
    pooled_: ConformalQuantileRegressor | None = None

    def fit(
        self, y_calib: np.ndarray, q_calib: np.ndarray, groups: np.ndarray
    ) -> MondrianConformalQuantileRegressor:
        self.pooled_ = ConformalQuantileRegressor(self.quantiles).fit(y_calib, q_calib)
        self.group_models_ = {}
        for g in pd.unique(groups):
            m = groups == g
            if m.sum() < self.min_group_n:
                continue
            self.group_models_[g] = ConformalQuantileRegressor(self.quantiles).fit(
                y_calib[m], q_calib[m]
            )
        log.info(
            "Mondrian CQR: %s groups calibrated individually, pooled fallback for the rest",
            len(self.group_models_),
        )
        return self

    def transform(self, q_pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
        if self.pooled_ is None:
            raise RuntimeError("call fit() before transform()")
        out = np.empty_like(q_pred)
        for g in pd.unique(groups):
            m = groups == g
            model = self.group_models_.get(g, self.pooled_)
            out[m] = model.transform(q_pred[m])
        return out


@dataclass
class AdaptiveConformalController:
    """Online adaptive conformal inference under distribution shift.

    Maintains a running miscoverage target ``alpha_t`` updated after every observation:

        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

    where ``err_t`` is 1 if the last interval missed. When the model starts undercovering
    the controller raises ``alpha_t``, which widens the interval; when it overcovers, the
    interval tightens again. The long-run empirical coverage converges to the target for
    *any* sequence, including adversarial ones -- exchangeability is not required, which
    is the whole point when a gas crisis rewrites the price distribution mid-sample.

    Parameters
    ----------
    alpha_target:
        Desired miscoverage rate, e.g. 0.1 for a 90% interval.
    gamma:
        Step size. Larger adapts faster but makes interval width noisier; 0.005-0.05 is
        the usual operating range.
    """

    alpha_target: float = 0.10
    gamma: float = 0.02
    alpha_t: float = field(init=False)
    history_: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.alpha_t = self.alpha_target

    def run(
        self,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        calib_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replay a test sequence, adapting the correction after each step.

        ``calib_scores`` seeds the score pool; it is extended online as truths arrive, so
        the controller both re-quantiles a growing sample and feeds back on realised
        misses.
        """
        scores = list(np.asarray(calib_scores, dtype=float))
        lo_out = np.empty_like(lower, dtype=float)
        hi_out = np.empty_like(upper, dtype=float)

        for t in range(len(y)):
            delta = _finite_sample_quantile(np.asarray(scores), self.alpha_t)
            lo_out[t] = lower[t] - delta
            hi_out[t] = upper[t] + delta

            covered = lo_out[t] <= y[t] <= hi_out[t]
            err = 0.0 if covered else 1.0
            self.history_.append(
                {"t": t, "alpha_t": self.alpha_t, "delta": delta, "covered": float(covered)}
            )
            # Feedback step, clipped to stay a valid miscoverage rate.
            self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha_target - err), 1e-3, 0.999))
            scores.append(float(max(lower[t] - y[t], y[t] - upper[t])))

        return lo_out, hi_out

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)


def hour_block_groups(index: pd.DatetimeIndex, tz: str = "Europe/Amsterdam") -> np.ndarray:
    """Group labels for Mondrian calibration: local time-of-day block crossed with season.

    Chosen because these are the axes along which forecast difficulty genuinely varies in
    the Dutch system -- winter evening peaks are hard, summer nights are easy -- so
    conditioning on them is where group-conditional coverage buys the most.
    """
    local = pd.DatetimeIndex(index).tz_convert(tz)
    block = pd.cut(
        local.hour,
        bins=[-1, 5, 9, 16, 20, 23],
        labels=["night", "morning_ramp", "midday", "evening_peak", "late"],
    ).astype(str)
    season = np.where(np.isin(local.month, [12, 1, 2]), "winter",
             np.where(np.isin(local.month, [3, 4, 5]), "spring",
             np.where(np.isin(local.month, [6, 7, 8]), "summer", "autumn")))
    return np.char.add(np.char.add(block.astype(str), "|"), season)
