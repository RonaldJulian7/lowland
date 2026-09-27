"""Build-out scenarios: what ~21 GW of offshore wind does to prices, curtailment,
capture rate and CO2.

Rather than assuming a price model, this learns the residual supply curve off eleven
years of data and re-evaluates it under counterfactual renewable output:

1. scale observed wind and solar by the capacity ratio. Scaling real output rather than
   simulating from weather keeps the actual spatial and temporal correlation structure,
   which a synthetic model would have to reinvent and would get subtly wrong;
2. recompute residual load, with an optional demand uplift for electrification applied on
   a peaked diurnal shape -- EV charging and heat demand aren't flat, and a flat uplift
   would understate exactly the evening hours that set scarcity;
3. run it back through the learned quantile supply curve;
4. curtail anything that would push residual load under the must-run floor.

The supply curve is predictive; the IV estimate in merit_order is causal. They answer
different questions, so both get reported -- if they disagree badly the scenario is
running on extrapolation and shouldn't be trusted. ScenarioResult.extrapolation_share
says how often a scenario leaves the residual-load range ever observed, and that number
belongs next to every result the scenario produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lowland.config import EMISSION_FACTORS_T_PER_MWH, QUANTILES
from lowland.utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Capacity assumptions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityScenario:
    """Installed-capacity multipliers relative to the observed base period.

    Multipliers, not absolute gigawatts, because the base period's effective capacity is
    implicit in the observed generation and is not cleanly recoverable from the feed --
    especially for solar, most of which is behind the meter and never appears as
    generation at all.
    """

    name: str
    wind_offshore_mult: float = 1.0
    wind_onshore_mult: float = 1.0
    solar_mult: float = 1.0
    #: Fractional change in annual electricity demand.
    demand_growth: float = 0.0
    #: Share of the demand growth that is peak-shaped (EV charging, heat pumps) rather
    #: than flat. 0 spreads it evenly; 1 concentrates it entirely in the evening peak.
    demand_peakiness: float = 0.6
    #: Battery/flex capacity able to shift energy, as a fraction of peak demand.
    storage_share_of_peak: float = 0.0
    description: str = ""


#: Scenarios anchored on published Dutch policy targets, expressed relative to the
#: 2024-2026 observed base.
SCENARIOS: tuple[CapacityScenario, ...] = (
    CapacityScenario("Base (observed)", 1.0, 1.0, 1.0, 0.0, description="Current system as observed"),
    CapacityScenario(
        "2030 offshore push", 2.4, 1.3, 1.6, 0.12, 0.6, 0.03,
        description="~21 GW offshore trajectory, continued solar growth, moderate electrification",
    ),
    CapacityScenario(
        "2035 high renewables", 3.6, 1.6, 2.4, 0.28, 0.65, 0.08,
        description="Deep decarbonisation pathway with substantial flexibility build-out",
    ),
    CapacityScenario(
        "2030 slow build", 1.6, 1.1, 1.3, 0.12, 0.6, 0.02,
        description="Grid and permitting delays hold offshore back",
    ),
)


# --------------------------------------------------------------------------------------
# Learned supply curve
# --------------------------------------------------------------------------------------


@dataclass
class SupplyCurveModel:
    """Learned mapping from residual load and context to the day-ahead price distribution.

    Quantile regression rather than a conditional mean, because the interesting
    counterfactual outcomes are distributional: how often the price goes negative, what
    the capture rate is, how wide the daily spread becomes.
    """

    quantiles: tuple[float, ...] = QUANTILES
    models_: dict = field(default_factory=dict)
    features_: list[str] = field(default_factory=list)
    support_: tuple[float, float] = (0.0, 0.0)
    #: Price sensitivity to residual load (EUR/MWh per GW) at each edge of the observed
    #: range, used to extend the curve beyond it.
    slope_lo_: float = 0.0
    slope_hi_: float = 0.0

    #: Renewable output deliberately does *not* appear here. Residual load already encodes
    #: the merit-order position (it is load minus renewables), so adding renewable level
    #: as a separate regressor is largely redundant inside the sample and actively harmful
    #: outside it: a counterfactual that triples offshore wind pushes that feature far
    #: beyond any observed value, where a tree model simply returns its boundary leaf.
    FEATURES = (
        "residual_load_gw", "load_gw",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
        "month_sin", "month_cos", "gas_regime", "year_frac",
    )

    @staticmethod
    def _design(
        residual_load: pd.Series,
        load: pd.Series,
        vre: pd.Series,
        index: pd.DatetimeIndex,
        gas_regime: pd.Series,
    ) -> pd.DataFrame:
        local = index.tz_convert("Europe/Amsterdam")
        return pd.DataFrame(
            {
                "residual_load_gw": residual_load.to_numpy() / 1000.0,
                "load_gw": load.to_numpy() / 1000.0,
                "vre_gw": vre.to_numpy() / 1000.0,
                "vre_share": (vre / load.replace(0, np.nan)).clip(0, 3).to_numpy(),
                "hour_sin": np.sin(2 * np.pi * local.hour / 24),
                "hour_cos": np.cos(2 * np.pi * local.hour / 24),
                "dow_sin": np.sin(2 * np.pi * local.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * local.dayofweek / 7),
                "is_weekend": (local.dayofweek >= 5).astype(float),
                "month_sin": np.sin(2 * np.pi * local.month / 12),
                "month_cos": np.cos(2 * np.pi * local.month / 12),
                "gas_regime": gas_regime.to_numpy(),
                "year_frac": (local.year - 2015) + local.dayofyear / 366.0,
            },
            index=index,
        )

    @staticmethod
    def gas_regime_proxy(panel: pd.DataFrame, window_days: int = 30) -> pd.Series:
        """A slow-moving proxy for marginal fuel cost.

        The feed contains no gas price. Since gas sets the Dutch marginal price in most
        hours, the rolling median price during *high* residual-load hours -- when a gas
        plant is almost certainly marginal -- tracks the fuel cost closely while being
        far less sensitive to renewable output than the raw price. Without a term like
        this the model would attribute the 2022 price level to that year's residual load,
        and every counterfactual would inherit the error.
        """
        price = panel["price_da"]
        rl = panel["residual_load"]
        high = rl > rl.quantile(0.75)
        peak_price = price.where(high)
        return (
            peak_price.rolling(f"{window_days}D", min_periods=24).median()
            .ffill().bfill()
        )

    def fit(self, panel: pd.DataFrame) -> SupplyCurveModel:
        import lightgbm as lgb

        df = panel
        if "is_provisional" in df.columns:
            df = df[df["is_provisional"] == 0]
        df = df.dropna(subset=["price_da", "residual_load", "load", "vre_total"])

        gas = self.gas_regime_proxy(panel).reindex(df.index)
        X = self._design(df["residual_load"], df["load"], df["vre_total"], df.index, gas)
        y = df["price_da"].clip(df["price_da"].quantile(0.001), df["price_da"].quantile(0.999))

        self.features_ = list(X.columns)
        self.support_ = (float(X["residual_load_gw"].min()), float(X["residual_load_gw"].max()))

        params = dict(
            objective="quantile", learning_rate=0.05, num_leaves=63, min_data_in_leaf=150,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, verbosity=-1, seed=7,
        )
        n_val = int(len(y) * 0.1)
        Xtr, ytr = X.iloc[:-n_val], y.iloc[:-n_val]
        Xva, yva = X.iloc[-n_val:], y.iloc[-n_val:]

        for q in self.quantiles:
            p = dict(params, alpha=q)
            self.models_[q] = lgb.train(
                p,
                lgb.Dataset(Xtr.to_numpy(), label=ytr.to_numpy(), feature_name=self.features_),
                num_boost_round=900,
                valid_sets=[lgb.Dataset(Xva.to_numpy(), label=yva.to_numpy())],
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
            )
        self._fit_boundary_slopes(X)
        log.info(
            "supply curve fitted on %s hours; residual-load support %.1f .. %.1f GW; "
            "edge slopes %.1f (low) / %.1f (high) EUR/MWh per GW",
            len(y), *self.support_, self.slope_lo_, self.slope_hi_,
        )
        return self

    def _fit_boundary_slopes(self, X: pd.DataFrame, edge_frac: float = 0.08) -> None:
        """Estimate the local price/residual-load slope at each edge of the sample.

        Gradient-boosted trees are piecewise constant, so beyond the range they were
        trained on they return a flat boundary value. For a counterfactual that deliberately
        pushes residual load outside that range, this is not a small inaccuracy: it makes
        the simulator report that tripling offshore wind barely moves the price, which is
        an artefact of the estimator rather than a finding about the power system.

        The fix is to extend the curve linearly beyond each edge using the slope the model
        itself exhibits *at* that edge, measured by regressing its own fitted values on
        residual load over the outermost ``edge_frac`` of the data. The slope is therefore
        still estimated from data, and -- importantly -- it is not the causal coefficient,
        so the cross-check in :func:`causal_cross_check` remains an independent comparison
        rather than a tautology.
        """
        rl = X["residual_load_gw"].to_numpy()
        med_i = int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))
        fitted = self.models_[self.quantiles[med_i]].predict(X[self.features_].to_numpy())

        lo_cut = np.quantile(rl, edge_frac)
        hi_cut = np.quantile(rl, 1 - edge_frac)

        def slope(mask: np.ndarray) -> float:
            if mask.sum() < 200:
                return 0.0
            a, b = rl[mask], fitted[mask]
            var = a.var()
            return float(np.cov(a, b)[0, 1] / var) if var > 1e-9 else 0.0

        self.slope_lo_ = slope(rl <= lo_cut)
        self.slope_hi_ = slope(rl >= hi_cut)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict price quantiles, extending linearly outside the observed support."""
        Xc = X[self.features_].copy()
        rl = Xc["residual_load_gw"].to_numpy()
        lo, hi = self.support_
        Xc["residual_load_gw"] = np.clip(rl, lo, hi)

        preds = np.column_stack(
            [self.models_[q].predict(Xc.to_numpy()) for q in self.quantiles]
        )

        # Parallel shift of the whole predictive distribution along the fitted edge slope.
        below = np.minimum(rl - lo, 0.0)
        above = np.maximum(rl - hi, 0.0)
        shift = below * self.slope_lo_ + above * self.slope_hi_
        preds = preds + shift[:, None]

        return np.sort(preds, axis=1)


# --------------------------------------------------------------------------------------
# Scenario evaluation
# --------------------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Outcome of one counterfactual scenario."""

    name: str
    hourly: pd.DataFrame
    metrics: dict[str, float]
    extrapolation_share: float
    description: str = ""


def _demand_shape(index: pd.DatetimeIndex, peakiness: float) -> np.ndarray:
    """Distribute demand growth across the day.

    A flat uplift understates the evening peak, which is where electrification actually
    bites: EV charging and heat-pump demand both concentrate between 17:00 and 21:00 local
    time. The shape is normalised to average one, so total annual growth is unaffected by
    the choice of ``peakiness``.
    """
    local = index.tz_convert("Europe/Amsterdam")
    h = local.hour.to_numpy()
    evening = np.exp(-0.5 * ((h - 19) / 3.0) ** 2)
    morning = 0.45 * np.exp(-0.5 * ((h - 8) / 2.5) ** 2)
    shape = 1.0 + peakiness * 2.2 * (evening + morning - (evening + morning).mean())
    return np.clip(shape, 0.2, None)


def simulate(
    panel: pd.DataFrame,
    model: SupplyCurveModel,
    scenario: CapacityScenario,
    *,
    base_years: tuple[int, ...] = (2024, 2025, 2026),
    must_run_gw: float = 1.8,
    price_floor: float = -500.0,
) -> ScenarioResult:
    """Evaluate one capacity scenario over a representative base period.

    Several recent years are used as the weather sample rather than one, so the result is
    not an artefact of a single unusually windy or still year -- interannual variability
    in Dutch wind output is on the order of 10%, which is large enough to swamp the
    difference between two of these scenarios.
    """
    df = panel
    if "is_provisional" in df.columns:
        df = df[df["is_provisional"] == 0]
    df = df.dropna(subset=["price_da", "residual_load", "load", "wind_offshore",
                           "wind_onshore", "solar"])
    local_years = df.index.tz_convert("Europe/Amsterdam").year
    df = df[np.isin(local_years, base_years)]
    if df.empty:
        raise ValueError("no observations in the requested base years")

    # --- counterfactual renewable output ------------------------------------------------
    wind_off = df["wind_offshore"] * scenario.wind_offshore_mult
    wind_on = df["wind_onshore"] * scenario.wind_onshore_mult
    solar = df["solar"] * scenario.solar_mult
    vre_potential = wind_off + wind_on + solar

    # --- counterfactual demand -----------------------------------------------------------
    shape = _demand_shape(df.index, scenario.demand_peakiness)
    load_cf = df["load"] * (1.0 + scenario.demand_growth * shape)

    # --- curtailment ----------------------------------------------------------------------
    # Thermal must-run (district heating, industrial CHP, nuclear) sets a floor below which
    # residual load cannot go; surplus renewables are curtailed instead.
    residual_raw = load_cf - vre_potential
    floor = must_run_gw * 1000.0
    curtailed = (floor - residual_raw).clip(lower=0.0)

    # --- storage absorbs part of the surplus -----------------------------------------------
    if scenario.storage_share_of_peak > 0:
        cap_mw = scenario.storage_share_of_peak * float(load_cf.max())
        # A daily energy budget: storage can absorb at most its power rating for a few
        # hours each day, so it does not magically soak up every surplus hour.
        absorbed = np.minimum(curtailed.to_numpy(), cap_mw)
        daily = pd.Series(absorbed, index=df.index).groupby(df.index.normalize()).cumsum()
        budget = cap_mw * 4.0
        absorbed = np.where(daily.to_numpy() <= budget, absorbed, 0.0)
        curtailed = curtailed - pd.Series(absorbed, index=df.index)
        curtailed = curtailed.clip(lower=0.0)

    vre_used = vre_potential - curtailed
    residual_cf = (load_cf - vre_used).clip(lower=floor)

    # --- price -------------------------------------------------------------------------------
    gas = model.gas_regime_proxy(panel).reindex(df.index)
    X_cf = model._design(residual_cf, load_cf, vre_used, df.index, gas)
    q_cf = model.predict(X_cf)
    q_cf = np.clip(q_cf, price_floor, None)

    med_i = int(np.argmin(np.abs(np.asarray(QUANTILES) - 0.5)))
    price_cf = q_cf[:, med_i]

    lo, hi = model.support_
    rl_gw = X_cf["residual_load_gw"].to_numpy()
    extrapolation = float(np.mean((rl_gw < lo) | (rl_gw > hi)))

    hourly = pd.DataFrame(
        {
            "load_mw": load_cf.to_numpy(),
            "vre_potential_mw": vre_potential.to_numpy(),
            "vre_used_mw": vre_used.to_numpy(),
            "curtailed_mw": curtailed.to_numpy(),
            "residual_load_mw": residual_cf.to_numpy(),
            "price_eur_mwh": price_cf,
            "price_p05": q_cf[:, 0],
            "price_p95": q_cf[:, -1],
            "price_observed": df["price_da"].to_numpy(),
        },
        index=df.index,
    )

    # --- metrics ---------------------------------------------------------------------------
    hours = len(hourly)
    wind_total = (wind_off + wind_on - curtailed * 0).sum()  # potential, before curtailment
    wind_used = float((wind_off + wind_on).sum()) - float(curtailed.sum())

    # Capture rate: the revenue a wind farm earns per MWh relative to the time-weighted
    # average price. This is the number that decides whether an unsubsidised project is
    # financeable, and it falls as penetration rises.
    wind_profile = (wind_off + wind_on).to_numpy()
    mean_price = float(np.mean(price_cf))
    capture = (
        float(np.sum(wind_profile * price_cf) / max(np.sum(wind_profile), 1e-6))
        if np.sum(wind_profile) > 0 else np.nan
    )

    thermal_mwh = float(np.maximum(residual_cf.to_numpy(), 0).sum())
    # Attribute residual load to the observed thermal mix, scaled to the counterfactual.
    base_thermal = float(df[["fossil_gas", "fossil_hard_coal"]].sum(axis=1).sum()) if (
        "fossil_gas" in df.columns
    ) else np.nan
    emissions = np.nan
    if np.isfinite(base_thermal) and base_thermal > 0:
        base_residual = float(np.maximum(df["residual_load"].to_numpy(), 0).sum())
        gas_share = float(df["fossil_gas"].sum()) / base_thermal
        coal_share = 1.0 - gas_share
        intensity = (
            gas_share * EMISSION_FACTORS_T_PER_MWH["Fossil gas"]
            + coal_share * EMISSION_FACTORS_T_PER_MWH["Fossil hard coal"]
        )
        # Scale by how much of residual load thermal plant actually served in the base.
        served_frac = base_thermal / max(base_residual, 1e-6)
        emissions = thermal_mwh * served_frac * intensity

    metrics = {
        "mean_price_eur_mwh": mean_price,
        "median_price_eur_mwh": float(np.median(price_cf)),
        "observed_mean_price_eur_mwh": float(df["price_da"].mean()),
        "negative_price_hours": int((price_cf < 0).sum()),
        "negative_price_share": float((price_cf < 0).mean()),
        "hours": hours,
        "mean_daily_spread_eur": float(
            pd.Series(price_cf, index=hourly.index).resample("D").apply(lambda s: s.max() - s.min()).mean()
        ),
        "vre_share_of_load": float(vre_used.sum() / load_cf.sum()),
        "curtailed_twh_per_year": float(curtailed.sum() / 1e6 * (8760 / max(hours, 1))),
        "curtailment_rate": float(curtailed.sum() / max(vre_potential.sum(), 1e-6)),
        "wind_capture_rate": capture / mean_price if mean_price else np.nan,
        "wind_capture_price_eur_mwh": capture,
        "residual_load_p95_mw": float(np.quantile(residual_cf, 0.95)),
        "co2_mt_per_year": (emissions / 1e6 * (8760 / max(hours, 1))) if np.isfinite(emissions) else np.nan,
    }

    return ScenarioResult(
        name=scenario.name,
        hourly=hourly,
        metrics=metrics,
        extrapolation_share=extrapolation,
        description=scenario.description,
    )


def causal_cross_check(
    base: ScenarioResult, scenario: ScenarioResult, causal_coef_eur_per_gw: float
) -> dict[str, float]:
    """Compare the structural scenario against a linear causal projection.

    Multiplies the IV-estimated merit-order coefficient by the change in mean renewable
    output. Agreement is evidence that the learned supply curve is behaving causally
    rather than exploiting a correlation; disagreement warns that the scenario has left
    the region the data can support.

    **The scenario passed here must hold demand fixed.** The IV coefficient is estimated
    with load as a control, so it answers "what does another GW of renewable output do,
    holding demand constant". A scenario that also grows demand by a quarter is answering
    a different question, and comparing the two conflates the merit-order effect with a
    demand effect pushing the other way. :mod:`projects.vantage.run` therefore
    builds a demand-neutral twin of each scenario purely for this check, and reports the
    demand-inclusive version separately as the policy-relevant result.

    Perfect agreement is not expected even so. The linear projection extrapolates a
    *marginal* effect across a large change, while the residual supply curve is convex:
    as residual load falls the system moves onto a flatter part of the merit order, so the
    per-GW impact diminishes. The structural estimate being smaller in magnitude than the
    linear one is the economically expected direction, and the ratio quantifies how much
    convexity the data implies.
    """
    d_vre_gw = (
        scenario.hourly["vre_used_mw"].mean() - base.hourly["vre_used_mw"].mean()
    ) / 1000.0
    linear = causal_coef_eur_per_gw * d_vre_gw
    structural = scenario.metrics["mean_price_eur_mwh"] - base.metrics["mean_price_eur_mwh"]
    return {
        "delta_vre_gw": float(d_vre_gw),
        "linear_causal_delta_eur": float(linear),
        "structural_model_delta_eur": float(structural),
        "divergence_eur": float(structural - linear),
        "agreement_ratio": float(structural / linear) if abs(linear) > 1e-6 else np.nan,
    }
