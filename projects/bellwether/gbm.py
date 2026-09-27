"""Gradient-boosted quantile regression with LightGBM.

One booster is trained per quantile level against the pinball objective. This is the
workhorse of applied energy forecasting: it handles the strong non-linearities
(temperature kinks, wind power curves, price spikes) and the mixed feature types without
scaling or imputation, and it trains in seconds rather than minutes.

Two details that matter and are easy to get wrong:

*Quantile crossing.* The levels are fitted independently, so nothing forces the 0.9
estimate above the 0.75 one. Crossings are counted and then removed by rearrangement
(sorting), which Chernozhukov, Fernandez-Val and Galichon showed cannot increase
estimation error.

*Early stopping needs a time-ordered validation split.* Using a random split would leak
future information into the stopping decision and systematically overfit, so the tail of
the training window is held out chronologically instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from lowland.config import QUANTILES, RANDOM_SEED
from lowland.metrics import quantile_crossing_rate
from lowland.utils import get_logger

log = get_logger(__name__)

#: Conservative defaults. Depth is capped and leaves kept modest because the sample is a
#: few tens of thousands of highly autocorrelated rows -- the effective sample size is far
#: smaller than the row count, so an unconstrained tree memorises weather noise.
DEFAULT_PARAMS: dict = {
    "objective": "quantile",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 64,
    "max_depth": 8,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "num_threads": 0,
}


@dataclass
class LightGBMQuantileForecaster:
    """A bank of LightGBM boosters, one per quantile level."""

    quantiles: tuple[float, ...] = QUANTILES
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    valid_frac: float = 0.12
    models_: dict[float, lgb.Booster] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    best_iters_: dict[float, int] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMQuantileForecaster:
        self.feature_names_ = list(X.columns)
        Xa = X.to_numpy(dtype=np.float32)
        ya = y.to_numpy(dtype=np.float64)

        ok = np.isfinite(ya)
        Xa, ya = Xa[ok], ya[ok]

        # Chronological hold-out: rows are already time-ordered, so the tail is the future.
        n_valid = max(500, int(len(ya) * self.valid_frac))
        n_valid = min(n_valid, len(ya) // 4)
        split = len(ya) - n_valid
        X_tr, y_tr = Xa[:split], ya[:split]
        X_va, y_va = Xa[split:], ya[split:]

        self.models_ = {}
        self.best_iters_ = {}
        for q in self.quantiles:
            params = dict(self.params)
            params["alpha"] = q
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names_)
            dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.models_[q] = booster
            self.best_iters_[q] = booster.best_iteration or self.num_boost_round

        log.info(
            "LightGBM fitted: n_train=%s n_valid=%s best_iters=%s",
            len(y_tr), len(y_va),
            {f"{q:.2f}": it for q, it in self.best_iters_.items()},
        )
        return self

    def predict_quantiles(self, X: pd.DataFrame, *, rearrange: bool = True) -> np.ndarray:
        Xa = X[self.feature_names_].to_numpy(dtype=np.float32)
        preds = np.column_stack(
            [self.models_[q].predict(Xa, num_iteration=self.best_iters_[q]) for q in self.quantiles]
        )
        rate = quantile_crossing_rate(preds)
        if rate > 0:
            log.debug("quantile crossings in %.2f%% of rows; rearranging", 100 * rate)
        return np.sort(preds, axis=1) if rearrange else preds

    # ---- interpretation -------------------------------------------------------------

    def importance(self, kind: str = "gain") -> pd.Series:
        """Feature importance of the median booster.

        Only the median model is used: the tail boosters answer a different question and
        averaging importances across quantile levels mixes them incoherently.
        """
        med = min(self.quantiles, key=lambda q: abs(q - 0.5))
        b = self.models_[med]
        imp = pd.Series(
            b.feature_importance(importance_type=kind), index=self.feature_names_
        ).sort_values(ascending=False)
        return imp / imp.sum() if imp.sum() else imp

    def grouped_importance(self) -> pd.Series:
        """Importance aggregated to feature families, which is what a reader can act on."""
        from lowland.features import feature_groups

        imp = self.importance()
        groups = feature_groups(self.feature_names_)
        return pd.Series(
            {g: float(imp.reindex(cols).fillna(0).sum()) for g, cols in groups.items()}
        ).sort_values(ascending=False)
