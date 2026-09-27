"""Causal identification of the merit-order effect.

How much does an extra GW of wind and solar knock off the day-ahead price? That number
decides whether further build-out cannibalises its own revenue.

Regressing price on renewable output gives a correlation, not the effect, and the biases
don't cancel. Demand moves both (cold still evenings: high demand, low wind, and the
naive fit charges wind for it). Curtailment reverses the causation when prices go deeply
negative. And the feed misses most behind-the-meter solar, so measurement error
attenuates on top.

Identification is weather: capacity-weighted hub-height wind speed and irradiance where
Dutch capacity actually is. Relevance is overwhelming and testable (output is ~cubic in
wind speed; first-stage F is reported). Exogeneity is about as literal as this gets --
price doesn't cause wind.

The exclusion restriction is the delicate part, since weather also reaches price through
demand. So the instruments are wind and irradiance only; temperature, degree-hours and
humidity are controls. Year-month fixed effects soak up fuel prices, which matters a lot
over a sample containing 2022.

Three estimators, because the disagreement between them is the informative part: naive
OLS (its gap from IV is the bias), 2SLS with Newey-West errors (hourly power residuals
are badly autocorrelated; classical SEs would overstate precision by an order of
magnitude), and cross-fitted DML so the answer doesn't hinge on having guessed the
functional form of the controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from lowland.utils import get_logger

log = get_logger(__name__)


@dataclass
class EffectEstimate:
    """One estimate of the merit-order effect, in EUR/MWh per GW of renewable output."""

    method: str
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int
    diagnostics: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "eur_per_mwh_per_gw": round(self.coefficient, 3),
            "std_error": round(self.std_error, 3),
            "t": round(self.t_stat, 2),
            "p_value": float(f"{self.p_value:.3g}"),
            "ci_low": round(self.ci_low, 3),
            "ci_high": round(self.ci_high, 3),
            "n": self.n,
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.diagnostics.items()},
        }


# --------------------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------------------

CONTROL_COLS = ("temp_pop", "hdh", "cdh", "humidity_pop", "temp_pop_ema24")
INSTRUMENT_COLS = ("wind_power_proxy_off", "wind_power_proxy_on", "wind100_off", "wind100_on", "ghi_pop")


def build_design(
    panel: pd.DataFrame,
    *,
    treatment: str = "vre_total",
    outcome: str = "price_da",
    drop_provisional: bool = True,
    winsorise: float = 0.001,
) -> dict[str, object]:
    """Assemble outcome, treatment, instruments, controls and fixed effects.

    The treatment is expressed in **GW** so the coefficient reads directly as
    "EUR/MWh per GW of additional renewable output".

    Extreme price outliers are winsorised at the 0.1% tails. Dutch day-ahead prices reach
    several hundred EUR/MWh in scarcity hours and once exceeded EUR 800; a handful of such
    observations would otherwise dominate a least-squares fit and turn the estimate into a
    statement about four afternoons in 2022.
    """
    df = panel.copy()
    if drop_provisional and "is_provisional" in df.columns:
        df = df[df["is_provisional"] == 0]
    if "is_complete" in df.columns:
        df = df[df["is_complete"] == 1]

    needed = [outcome, treatment, *CONTROL_COLS, *INSTRUMENT_COLS, "load"]
    needed = [c for c in needed if c in df.columns]
    df = df.dropna(subset=needed)

    y = df[outcome].astype(float)
    if winsorise > 0:
        lo, hi = y.quantile(winsorise), y.quantile(1 - winsorise)
        y = y.clip(lo, hi)

    d = df[treatment].astype(float) / 1000.0  # MW -> GW

    z_cols = [c for c in INSTRUMENT_COLS if c in df.columns]
    Z = df[z_cols].astype(float)
    # Wind power is cubic in wind speed below rated; including the square of the capacity
    # factor lets the first stage capture the saturation without a functional-form guess.
    Z = Z.assign(
        wind_off_sq=df["wind_power_proxy_off"] ** 2,
        wind_on_sq=df["wind_power_proxy_on"] ** 2,
        ghi_sq=(df["ghi_pop"] / 1000.0) ** 2,
    )

    x_cols = [c for c in CONTROL_COLS if c in df.columns]
    X = df[x_cols].astype(float)
    # Demand is a confounder, not a mediator: it moves price directly and is correlated
    # with weather. Controlling for it is what isolates the supply-side channel.
    X = X.assign(load_gw=df["load"] / 1000.0)

    local = df.index.tz_convert("Europe/Amsterdam")
    fe = pd.DataFrame(
        {
            "hour": local.hour,
            "dow": local.dayofweek,
            "ym": local.year * 100 + local.month,
        },
        index=df.index,
    )

    return {"y": y, "d": d, "Z": Z, "X": X, "fe": fe, "index": df.index, "frame": df}


def _dummies(fe: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the fixed effects, dropping one level of each to avoid collinearity."""
    parts = [pd.get_dummies(fe[c].astype("category"), prefix=c, drop_first=True) for c in fe.columns]
    return pd.concat(parts, axis=1).astype(float)


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of the moment condition ``X'e`` with a Bartlett kernel."""
    n, k = X.shape
    u = X * resid[:, None]
    S = u.T @ u / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag] / n
        S += w * (G + G.T)
    return S * n


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------


def ols_effect(design: dict, *, hac_lags: int = 48) -> EffectEstimate:
    """Naive OLS of price on renewable output with controls and fixed effects."""
    y = design["y"].to_numpy()
    W = np.column_stack(
        [
            np.ones(len(y)),
            design["d"].to_numpy(),
            design["X"].to_numpy(),
            _dummies(design["fe"]).to_numpy(),
        ]
    )
    XtX_inv = np.linalg.pinv(W.T @ W)
    beta = XtX_inv @ W.T @ y
    resid = y - W @ beta

    S = _newey_west_cov(W, resid, hac_lags)
    cov = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(cov[1, 1], 0)))
    b = float(beta[1])
    t = b / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    return EffectEstimate(
        "OLS (naive)", b, se, t, p, b - 1.96 * se, b + 1.96 * se, len(y),
        {"r2": float(1 - resid.var() / y.var())},
    )


def iv_2sls(design: dict, *, hac_lags: int = 48) -> EffectEstimate:
    """Two-stage least squares with weather instruments and Newey-West standard errors."""
    y = design["y"].to_numpy()
    d = design["d"].to_numpy()
    X = design["X"].to_numpy()
    Zi = design["Z"].to_numpy()
    D = _dummies(design["fe"]).to_numpy()

    exog = np.column_stack([np.ones(len(y)), X, D])
    W = np.column_stack([d[:, None], exog])          # endogenous + exogenous regressors
    Zfull = np.column_stack([Zi, exog])              # instruments + exogenous regressors

    # First stage, for the relevance diagnostic.
    ZtZ_inv = np.linalg.pinv(Zfull.T @ Zfull)
    pi = ZtZ_inv @ Zfull.T @ d
    d_hat = Zfull @ pi
    resid_1 = d - d_hat
    # Partial (Kleibergen-Paap style) F on the excluded instruments only.
    exog_inv = np.linalg.pinv(exog.T @ exog)
    d_on_exog = exog @ (exog_inv @ exog.T @ d)
    rss_restricted = float(((d - d_on_exog) ** 2).sum())
    rss_full = float((resid_1**2).sum())
    q = Zi.shape[1]
    dof = len(y) - Zfull.shape[1]
    first_stage_f = ((rss_restricted - rss_full) / q) / (rss_full / dof) if rss_full > 0 else np.nan

    # 2SLS via the projection matrix.
    PW = Zfull @ (ZtZ_inv @ (Zfull.T @ W))
    A = np.linalg.pinv(W.T @ PW)
    beta = A @ (PW.T @ y)
    resid = y - W @ beta

    S = _newey_west_cov(PW, resid, hac_lags)
    cov = A @ S @ A
    se = float(np.sqrt(max(cov[0, 0], 0)))
    b = float(beta[0])
    t = b / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t)))

    # Sargan overidentification test: with more instruments than endogenous regressors,
    # the model's own exclusion restrictions become testable.
    u = resid
    theta = ZtZ_inv @ (Zfull.T @ u)
    sargan = float(len(y) * (u @ Zfull @ theta) / (u @ u)) if (u @ u) > 0 else np.nan
    sargan_df = max(q - 1, 1)
    sargan_p = float(1 - stats.chi2.cdf(sargan, sargan_df)) if np.isfinite(sargan) else np.nan

    return EffectEstimate(
        "IV-2SLS (weather instruments)", b, se, t, p, b - 1.96 * se, b + 1.96 * se, len(y),
        {
            "first_stage_F": float(first_stage_f),
            "n_instruments": int(q),
            "sargan_stat": sargan,
            "sargan_p": sargan_p,
            "hac_lags": hac_lags,
        },
    )


def dml_pliv(
    design: dict,
    *,
    n_folds: int = 5,
    hac_lags: int = 48,
    seed: int = 20260924,
) -> EffectEstimate:
    """Double/debiased ML for the partially linear IV model.

    Estimates the nuisance functions ``E[Y|X]``, ``E[D|X]`` and ``E[Z|X]`` with gradient
    boosting, then forms the orthogonal (Neyman-orthogonal) score

        theta = E[(Z - E[Z|X])(Y - E[Y|X])] / E[(Z - E[Z|X])(D - E[D|X])]

    Cross-fitting means every residual is computed by a model that never saw that
    observation, which is what makes the estimator root-n consistent despite the nuisance
    functions converging more slowly. Folds are contiguous blocks in time rather than
    random, so the autocorrelation in the data cannot leak a hold-out observation's value
    into the model that predicts it.
    """
    import lightgbm as lgb

    y = design["y"].to_numpy()
    d = design["d"].to_numpy()

    Xc = pd.concat([design["X"], design["fe"].astype(float)], axis=1).to_numpy()
    # Collapse the instrument set to a single scalar via its first-stage projection: the
    # PLIV score is defined for a scalar instrument, and the optimal combination is the
    # fitted value from regressing D on Z.
    Zi = design["Z"].to_numpy()
    # Parenthesised right-to-left on purpose. Left-to-right evaluation would materialise
    # Zi @ pinv(...) as an (n x n) matrix -- 77 GiB at this sample size -- before ever
    # touching d. Grouping the small factors first keeps every intermediate (k x k) or
    # (k x 1).
    z_proj = Zi @ (np.linalg.pinv(Zi.T @ Zi) @ (Zi.T @ d))

    n = len(y)
    folds = np.array_split(np.arange(n), n_folds)  # contiguous in time
    ry, rd, rz = np.zeros(n), np.zeros(n), np.zeros(n)

    params = dict(
        objective="regression", learning_rate=0.06, num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=seed,
    )

    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        for target, out in ((y, ry), (d, rd), (z_proj, rz)):
            model = lgb.train(
                params,
                lgb.Dataset(Xc[train_idx], label=target[train_idx]),
                num_boost_round=300,
            )
            out[test_idx] = target[test_idx] - model.predict(Xc[test_idx])
        log.debug("DML fold %s/%s done", k + 1, n_folds)

    num = float(np.mean(rz * ry))
    den = float(np.mean(rz * rd))
    theta = num / den if den != 0 else np.nan

    # Influence-function standard error with a HAC correction, matching the IV estimator.
    psi = rz * (ry - theta * rd)
    J = -den
    infl = psi / J
    S = _newey_west_cov(infl[:, None], np.ones(n), hac_lags)[0, 0] / n
    se = float(np.sqrt(max(S / n, 0)))
    t = theta / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t)))

    return EffectEstimate(
        "DML-PLIV (cross-fitted)", float(theta), se, float(t), float(p),
        theta - 1.96 * se, theta + 1.96 * se, n,
        {"n_folds": n_folds, "first_stage_partial_corr": float(np.corrcoef(rz, rd)[0, 1])},
    )


# --------------------------------------------------------------------------------------
# Heterogeneity
# --------------------------------------------------------------------------------------


def heterogeneous_effects(
    panel: pd.DataFrame, by: str = "year", **kwargs
) -> pd.DataFrame:
    """Re-estimate the IV effect within subgroups.

    The merit-order effect is not a constant of nature: it is the slope of the residual
    supply curve, which steepens as thermal plant retires and flattens when storage and
    interconnection absorb the surplus. Estimating it by year is how that structural
    change becomes visible, and it is the part that matters for a 2030 projection.
    """
    rows = []
    local = panel.index.tz_convert("Europe/Amsterdam")
    if by == "year":
        groups = local.year
    elif by == "season":
        groups = np.where(np.isin(local.month, [12, 1, 2]), "winter",
                 np.where(np.isin(local.month, [6, 7, 8]), "summer", "shoulder"))
    elif by == "peak":
        groups = np.where((local.hour >= 7) & (local.hour < 20), "peak", "off-peak")
    else:
        raise ValueError(f"unknown grouping {by!r}")

    for g in pd.unique(groups):
        sub = panel[groups == g]
        if len(sub) < 2000:
            continue
        try:
            design = build_design(sub, **kwargs)
            if len(design["y"]) < 1000:
                continue
            est = iv_2sls(design)
            row = est.as_row()
            row[by] = g
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - a degenerate subgroup must not abort the sweep
            log.warning("subgroup %s failed: %s", g, exc)

    df = pd.DataFrame(rows)
    return df.sort_values(by).reset_index(drop=True) if len(df) else df


def run_all(panel: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """Estimate the effect by all three methods and return a comparison table."""
    design = build_design(panel, **kwargs)
    estimates = [ols_effect(design), iv_2sls(design), dml_pliv(design)]
    table = pd.DataFrame([e.as_row() for e in estimates])

    iv = estimates[1]
    ols = estimates[0]
    context = {
        "n_hours": int(len(design["y"])),
        "sample_start": str(design["index"].min()),
        "sample_end": str(design["index"].max()),
        "mean_price_eur_mwh": float(design["y"].mean()),
        "mean_vre_gw": float(design["d"].mean()),
        "ols_iv_gap": round(ols.coefficient - iv.coefficient, 3),
        "first_stage_F": iv.diagnostics.get("first_stage_F"),
    }
    return table, context
