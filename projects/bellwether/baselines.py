"""Probabilistic baselines.

All three emit a full predictive distribution, not a point. Comparing a quantile model's
CRPS against a point baseline's MAE proves nothing.

In increasing order of strength: climatology (knows the hour and the month and nothing
else -- fail to beat it and you've learned nothing), seasonal naive (same hour last week
plus an empirical residual distribution; deceptively strong, Dutch demand repeats weekly),
and a linear quantile AR with no weather, which is what separates "our model is good" from
"our model discovered that Tuesdays look like Tuesdays".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lowland.config import QUANTILES
from lowland.utils import get_logger

log = get_logger(__name__)


@dataclass
class ClimatologyForecaster:
    """Empirical quantiles conditional on (hour of day, month).

    Deliberately ignores every predictor except the calendar. Its CRPS is the score to
    beat before any modelling claim is credible.
    """

    quantiles: tuple[float, ...] = QUANTILES
    table_: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    global_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClimatologyForecaster:
        df = pd.DataFrame({"y": y.to_numpy(), "hour": X["hour"].to_numpy(), "month": X["month"].to_numpy()})
        df = df.dropna()
        self.global_ = np.quantile(df["y"], self.quantiles)
        self.table_ = {}
        for (h, m), grp in df.groupby(["hour", "month"]):
            if len(grp) >= 30:  # below this the tail quantiles are pure noise
                self.table_[(int(h), int(m))] = np.quantile(grp["y"], self.quantiles)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        assert self.global_ is not None, "call fit() first"
        out = np.tile(self.global_, (len(X), 1))
        hours = X["hour"].to_numpy(dtype=int)
        months = X["month"].to_numpy(dtype=int)
        for i, (h, m) in enumerate(zip(hours, months)):
            q = self.table_.get((h, m))
            if q is not None:
                out[i] = q
        return out


@dataclass
class SeasonalNaiveForecaster:
    """Same hour last week, plus an hour-of-day-conditional empirical error distribution.

    The weekly rather than daily lag is deliberate: it preserves the weekday/weekend
    pattern, which a 24 h lag destroys every Saturday and Monday.
    """

    quantiles: tuple[float, ...] = QUANTILES
    lag_col: str = "y_slag168"
    resid_q_: dict[int, np.ndarray] = field(default_factory=dict)
    global_resid_q_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaiveForecaster:
        if self.lag_col not in X.columns:
            raise KeyError(f"{self.lag_col} missing; SeasonalNaive needs the weekly lag feature")
        resid = y.to_numpy() - X[self.lag_col].to_numpy()
        hours = X["hour"].to_numpy(dtype=int)
        ok = np.isfinite(resid)
        self.global_resid_q_ = np.quantile(resid[ok], self.quantiles)
        self.resid_q_ = {}
        for h in range(24):
            m = ok & (hours == h)
            if m.sum() >= 30:
                self.resid_q_[h] = np.quantile(resid[m], self.quantiles)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        assert self.global_resid_q_ is not None, "call fit() first"
        base = X[self.lag_col].to_numpy(dtype=float)
        hours = X["hour"].to_numpy(dtype=int)
        out = np.empty((len(X), len(self.quantiles)))
        for i, (b, h) in enumerate(zip(base, hours)):
            out[i] = b + self.resid_q_.get(h, self.global_resid_q_)
        # A missing lag leaves nothing to anchor on; fall back to the marginal spread
        # centred on the training mean rather than propagating NaN into the metrics.
        bad = ~np.isfinite(base)
        if bad.any():
            out[bad] = np.nanmean(base) + self.global_resid_q_
        return out


@dataclass
class QuantileARForecaster:
    """Linear quantile regression on calendar + autoregressive features only.

    Weather is withheld on purpose. The gap between this and the full models is a clean
    read on how much the meteorology contributes, which is the question an energy
    forecaster is actually asked.

    Fitted by direct gradient descent on the pinball loss rather than by linear
    programming (scikit-learn's ``QuantileRegressor``) or IRLS (statsmodels' ``QuantReg``).
    Both of those were tried first and both failed on this problem: the LP does not finish
    in usable time at ~10^5 rows, and IRLS does not converge on a design matrix this
    collinear -- it returned coefficients two orders of magnitude out, producing a
    "baseline" with an MAE of 294 GW. Minimising the pinball loss directly is convex,
    converges monotonically, fits all quantile levels in one pass, and cannot diverge.
    """

    quantiles: tuple[float, ...] = QUANTILES
    n_steps: int = 600
    lr: float = 0.05
    l2: float = 1e-4
    models_: np.ndarray | None = None   # (n_features + 1, n_quantiles)
    cols_: list[str] = field(default_factory=list)
    means_: pd.Series | None = None
    scale_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileARForecaster:
        import torch

        from lowland.features import feature_groups
        from lowland.utils import resolve_device

        groups = feature_groups(list(X.columns))
        keep = groups.get("calendar", []) + groups.get("seasonality", []) + groups.get(
            "autoregressive", []
        )
        self.cols_ = [c for c in keep if c in X.columns]

        Xf = X[self.cols_]
        self.means_ = Xf.mean()
        self.scale_ = Xf.std().replace(0, 1.0).fillna(1.0)
        Xs = ((Xf - self.means_) / self.scale_).fillna(0.0)

        ok = np.isfinite(y.to_numpy()) & np.isfinite(Xs.to_numpy()).all(axis=1)
        Xa = np.c_[np.ones(ok.sum()), Xs.to_numpy()[ok]].astype(np.float32)
        ya = y.to_numpy()[ok].astype(np.float32)

        # Centre and scale the target too, so one learning rate works for both MW-scale
        # load and EUR/MWh-scale price without retuning.
        y_mu, y_sd = float(ya.mean()), float(ya.std() + 1e-6)
        yn = (ya - y_mu) / y_sd

        dev = resolve_device()
        Xt = torch.from_numpy(Xa).to(dev)
        yt = torch.from_numpy(yn).to(dev).unsqueeze(1)
        taus = torch.tensor(self.quantiles, device=dev, dtype=torch.float32).view(1, -1)

        beta = torch.zeros(Xa.shape[1], len(self.quantiles), device=dev, requires_grad=True)
        with torch.no_grad():
            # Start each level at its unconditional quantile; the intercept is then
            # already right and the optimiser only has to learn the slopes.
            beta[0] = torch.tensor(
                np.quantile(yn, self.quantiles), device=dev, dtype=torch.float32
            )

        opt = torch.optim.Adam([beta], lr=self.lr)
        for _ in range(self.n_steps):
            opt.zero_grad(set_to_none=True)
            pred = Xt @ beta
            d = yt - pred
            loss = torch.maximum(taus * d, (taus - 1.0) * d).mean()
            loss = loss + self.l2 * beta[1:].pow(2).mean()
            loss.backward()
            opt.step()

        b = beta.detach().cpu().numpy()
        # Undo the target scaling so predict() works in original units.
        b = b * y_sd
        b[0] += y_mu
        self.models_ = b
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        assert self.models_ is not None, "call fit() first"
        Xs = ((X[self.cols_] - self.means_) / self.scale_).fillna(0.0)
        Xa = np.c_[np.ones(len(Xs)), Xs.to_numpy()]
        return np.sort(Xa @ self.models_, axis=1)
