"""Correctness tests for the properties that would silently invalidate results.

These are not coverage-chasing unit tests. Each one checks a property whose violation
would produce numbers that *look* fine and are wrong -- the failure mode that matters in
empirical work, because nothing crashes and the plots still render.

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lowland.backtest import RollingOriginSplitter, calibration_split
from lowland.conformal import ConformalQuantileRegressor, cqr_scores
from lowland.features import FeatureSpec, make_supervised
from lowland.metrics import (
    coverage,
    crps_from_quantiles,
    diebold_mariano,
    interval_score,
    pinball_loss,
    rearrange_quantiles,
)

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    """A small panel with known structure, for leakage and shape tests."""
    idx = pd.date_range("2022-01-01", periods=24 * 400, freq="h", tz="UTC")
    n = len(idx)
    t = np.arange(n)
    daily = 2000 * np.sin(2 * np.pi * t / 24)
    weekly = 800 * np.sin(2 * np.pi * t / 168)
    temp = 10 + 8 * np.sin(2 * np.pi * t / 8766) + RNG.normal(0, 2, n)
    load = 11000 + daily + weekly - 250 * (temp - 15) + RNG.normal(0, 300, n)
    wind = np.clip(RNG.gamma(2.0, 900, n), 0, 6000)
    solar = np.clip(1500 * np.sin(np.pi * ((t % 24) - 6) / 12), 0, None)

    df = pd.DataFrame(
        {
            "load": load,
            "wind_offshore": wind * 0.6,
            "wind_onshore": wind * 0.4,
            "solar": solar,
            "vre_total": wind + solar,
            "residual_load": load - wind - solar,
            "price_da": 60 + 0.004 * (load - wind - solar) + RNG.normal(0, 8, n),
            "temp_pop": temp,
            "hdh": np.clip(15.5 - temp, 0, None),
            "cdh": np.clip(temp - 22, 0, None),
            "temp_pop_ema24": pd.Series(temp, index=idx).ewm(halflife=24).mean().to_numpy(),
            "hdh_ema24": pd.Series(np.clip(15.5 - temp, 0, None), index=idx).ewm(halflife=24).mean().to_numpy(),
            "humidity_pop": RNG.uniform(60, 95, n),
            "wind100_off": RNG.uniform(2, 20, n),
            "wind100_on": RNG.uniform(2, 18, n),
            "wind_power_proxy_off": RNG.uniform(0, 1, n),
            "wind_power_proxy_on": RNG.uniform(0, 1, n),
            "ghi_pop": np.clip(solar / 3, 0, None),
            "dni_pop": np.clip(solar / 4, 0, None),
            "cloud_pop": RNG.uniform(0, 100, n),
            "solar_proxy": np.clip(solar / 3, 0, None),
            "is_complete": 1,
            "is_provisional": 0,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


# --------------------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_autoregressive_features_never_see_past_the_origin(synthetic_panel, horizon):
    """Every autoregressive feature must be computable at ``target_time - horizon``.

    This is the single most consequential property in the repository. A lag shorter than
    the horizon lets the model observe the outcome it is predicting, which produces
    spectacular backtest scores and a system that fails the moment it is deployed.

    The test reconstructs each lag feature from the raw series and asserts that the value
    attached to time ``t`` equals the series at ``t - k`` for some ``k >= horizon``.
    """
    spec = FeatureSpec(target="residual_load", horizon=horizon)
    X, y, origins = make_supervised(synthetic_panel, spec)
    series = synthetic_panel["residual_load"]

    lag_cols = [c for c in X.columns if c.startswith("y_lag") or c.startswith("y_slag")]
    assert lag_cols, "expected autoregressive lag features to exist"

    for col in lag_cols:
        k = int(col.replace("y_lag", "").replace("y_slag", ""))
        assert k >= horizon, f"{col} uses lag {k}, shorter than the {horizon}h horizon"
        expected = series.shift(k)
        ok = X[col].notna() & expected.notna()
        assert ok.sum() > 100
        np.testing.assert_allclose(
            X.loc[ok, col].to_numpy(), expected[ok].to_numpy(), rtol=1e-9,
            err_msg=f"{col} does not equal the series shifted by {k}",
        )

    # The origin index must be exactly one horizon before the target.
    assert (X.index - origins == pd.Timedelta(hours=horizon)).all()


def test_rolling_features_do_not_include_the_target_hour(synthetic_panel):
    """Rolling statistics must be computed strictly on data available at the origin."""
    horizon = 24
    spec = FeatureSpec(target="residual_load", horizon=horizon)
    X, _, _ = make_supervised(synthetic_panel, spec)
    series = synthetic_panel["residual_load"]

    col = "y_rmean24"
    assert col in X.columns
    expected = series.shift(horizon).rolling(24, min_periods=6).mean()
    ok = X[col].notna() & expected.notna()
    np.testing.assert_allclose(
        X.loc[ok, col].to_numpy(), expected[ok].to_numpy(), rtol=1e-9
    )


def test_backtest_purges_the_horizon_between_train_and_test(synthetic_panel):
    """No training origin may have its target inside the test window."""
    horizon = 24
    spec = FeatureSpec(target="residual_load", horizon=horizon)
    _, _, origins = make_supervised(synthetic_panel, spec)

    splitter = RollingOriginSplitter(
        n_folds=3, test_days=20, horizon=horizon, min_train_days=120
    )
    folds = list(splitter.split(origins))
    assert folds, "expected at least one fold"

    for fold in folds:
        train_origins = origins[fold.train_mask]
        test_origins = origins[fold.test_mask]
        assert train_origins.max() < test_origins.min()
        # The target of the last training row lands at origin + horizon; it must still
        # fall strictly before the first scored target.
        last_train_target = train_origins.max() + pd.Timedelta(hours=horizon)
        first_test_target = test_origins.min() + pd.Timedelta(hours=horizon)
        assert last_train_target <= first_test_target


def test_calibration_block_is_disjoint_from_the_fitting_block(synthetic_panel):
    """Conformal calibration is only valid on data the model has not been fitted on."""
    horizon = 24
    spec = FeatureSpec(target="residual_load", horizon=horizon)
    _, _, origins = make_supervised(synthetic_panel, spec)
    splitter = RollingOriginSplitter(n_folds=2, test_days=20, horizon=horizon, min_train_days=120)
    fold = next(iter(splitter.split(origins)))

    fit_mask, calib_mask = calibration_split(fold.train_mask, origins, calib_days=60, horizon=horizon)
    assert calib_mask.sum() > 0
    assert not (fit_mask & calib_mask).any(), "fitting and calibration blocks overlap"
    assert origins[fit_mask].max() < origins[calib_mask].min()


# --------------------------------------------------------------------------------------
# Conformal prediction
# --------------------------------------------------------------------------------------


def test_conformal_delivers_nominal_coverage_on_exchangeable_data():
    """The core CQR guarantee: coverage at least 1 - alpha, whatever the base model.

    A deliberately overconfident base model (intervals far too narrow) is calibrated on a
    held-out block; the corrected intervals must then reach nominal coverage on fresh data
    drawn from the same distribution.
    """
    n = 4000
    y_all = RNG.normal(100, 25, n)
    # A badly miscalibrated model: it reports a spread of 5 when the truth is 25.
    z = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])
    q_all = 100 + np.outer(np.ones(n), z * 5.0)

    calib, test = slice(0, 2000), slice(2000, n)
    before = coverage(y_all[test], q_all[test][:, 0], q_all[test][:, -1])
    assert before < 0.5, "the base model should be badly overconfident to start with"

    cqr = ConformalQuantileRegressor(QUANTILES).fit(y_all[calib], q_all[calib])
    q_cal = cqr.transform(q_all[test])
    after = coverage(y_all[test], q_cal[:, 0], q_cal[:, -1])

    assert after >= 0.88, f"conformal coverage {after:.3f} fell short of the 0.90 target"
    assert after <= 0.98, "conformal intervals should not be wildly conservative either"


def test_conformal_tightens_an_over_wide_interval():
    """CQR must be able to shrink intervals, not only widen them.

    The signed non-conformity score is what makes this possible; an implementation using
    the absolute residual would only ever inflate.
    """
    n = 4000
    y_all = RNG.normal(50, 5, n)
    z = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])
    q_all = 50 + np.outer(np.ones(n), z * 40.0)  # far too wide

    calib, test = slice(0, 2000), slice(2000, n)
    width_before = float(np.mean(q_all[test][:, -1] - q_all[test][:, 0]))

    cqr = ConformalQuantileRegressor(QUANTILES).fit(y_all[calib], q_all[calib])
    q_cal = cqr.transform(q_all[test])
    width_after = float(np.mean(q_cal[:, -1] - q_cal[:, 0]))

    assert width_after < width_before * 0.5
    assert coverage(y_all[test], q_cal[:, 0], q_cal[:, -1]) >= 0.86


def test_cqr_scores_sign_convention():
    """Positive when the truth escapes the interval, negative when it sits inside."""
    y = np.array([5.0, 15.0, 0.0])
    lo = np.array([0.0, 0.0, 10.0])
    hi = np.array([10.0, 10.0, 20.0])
    s = cqr_scores(y, lo, hi)
    assert s[0] < 0            # inside, with slack
    assert s[1] == pytest.approx(5.0)   # above the upper bound by 5
    assert s[2] == pytest.approx(10.0)  # below the lower bound by 10


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_pinball_loss_is_minimised_at_the_true_quantile():
    """The defining property of a proper scoring rule for quantile estimation."""
    y = RNG.normal(0, 1, 60000)
    for tau in (0.1, 0.5, 0.9):
        true_q = float(np.quantile(y, tau))
        best = pinball_loss(y, np.full_like(y, true_q), tau)
        for offset in (-0.4, -0.15, 0.15, 0.4):
            worse = pinball_loss(y, np.full_like(y, true_q + offset), tau)
            assert worse > best, f"pinball at tau={tau} not minimised at the true quantile"


def test_crps_rewards_the_sharper_of_two_calibrated_forecasts():
    """Between two calibrated forecasts, CRPS must prefer the one with less spread."""
    y = RNG.normal(0, 1, 20000)
    z = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])
    tight = np.outer(np.ones(len(y)), z * 1.0)
    loose = np.outer(np.ones(len(y)), z * 3.0)
    assert crps_from_quantiles(y, tight, QUANTILES) < crps_from_quantiles(y, loose, QUANTILES)


def test_interval_score_penalises_miscoverage_more_than_width():
    """A narrow interval that misses must score worse than a wider one that covers."""
    y = np.full(1000, 10.0)
    covering = interval_score(y, np.full(1000, 5.0), np.full(1000, 15.0), alpha=0.1)
    missing = interval_score(y, np.full(1000, 0.0), np.full(1000, 2.0), alpha=0.1)
    assert missing > covering


def test_rearrangement_fixes_crossing_without_changing_the_set_of_values():
    q = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    out = rearrange_quantiles(q)
    assert (np.diff(out, axis=1) >= 0).all()
    np.testing.assert_allclose(np.sort(q, axis=1), out)


def test_diebold_mariano_detects_a_real_difference_and_ignores_a_null():
    """Significant when one model is genuinely better; not significant when they tie."""
    n = 3000
    a = RNG.normal(1.0, 0.5, n)
    b = RNG.normal(1.5, 0.5, n)   # model B is worse
    stat, p = diebold_mariano(a, b, h=1)
    assert stat < 0 and p < 0.01

    c = RNG.normal(1.0, 0.5, n)
    d = RNG.normal(1.0, 0.5, n)
    _, p_null = diebold_mariano(c, d, h=1)
    assert p_null > 0.05


# --------------------------------------------------------------------------------------
# Congestion risk
# --------------------------------------------------------------------------------------


def test_exceedance_probability_is_monotone_and_matches_the_quantiles():
    """P(Y > t) must fall as t rises, and equal 1 - tau at the estimated quantiles."""
    from projects.bellwether.congestion import cdf_at, exceedance_probability

    q = np.array([[80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]])
    thresholds = np.linspace(60, 160, 40)
    probs = [float(exceedance_probability(q, QUANTILES, t)[0]) for t in thresholds]
    assert all(b <= a + 1e-9 for a, b in zip(probs, probs[1:])), "exceedance must be monotone"

    for i, tau in enumerate(QUANTILES):
        assert cdf_at(q, QUANTILES, q[0, i])[0] == pytest.approx(tau, abs=1e-6)


def test_expected_exceedance_is_zero_far_above_the_distribution():
    from projects.bellwether.congestion import expected_exceedance

    q = np.array([[80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]])
    assert expected_exceedance(q, QUANTILES, 1e4)[0] == pytest.approx(0.0, abs=1e-6)
    assert expected_exceedance(q, QUANTILES, 60.0)[0] > 40.0


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def test_milp_schedule_respects_the_physics():
    """State of charge must stay within bounds and follow the efficiency dynamics exactly."""
    from lowland.config import BatterySpec
    from projects.flywheel.dispatch import solve_perfect_foresight

    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    prices = pd.Series(60 + 40 * np.sin(2 * np.pi * np.arange(48) / 24), index=idx)
    spec = BatterySpec(power_mw=10, energy_mwh=20, degradation_cost_eur_per_mwh=1.0)

    res = solve_perfect_foresight(prices, spec)
    s = res.schedule

    assert (s["soc_mwh"] >= spec.soc_min - 1e-6).all()
    assert (s["soc_mwh"] <= spec.soc_max + 1e-6).all()
    assert (s["charge_mw"] <= spec.power_mw + 1e-6).all()
    assert (s["discharge_mw"] <= spec.power_mw + 1e-6).all()
    # Never charging and discharging in the same hour, which the binaries enforce.
    assert ((s["charge_mw"] > 1e-6) & (s["discharge_mw"] > 1e-6)).sum() == 0

    soc = spec.soc_init
    for _, row in s.iterrows():
        soc = soc + spec.eta_charge * row["charge_mw"] - row["discharge_mw"] / spec.eta_discharge
        assert row["soc_mwh"] == pytest.approx(soc, abs=1e-5)


def test_perfect_foresight_bounds_every_other_controller():
    """The MILP optimum must not be beaten by a heuristic on the same prices."""
    from lowland.config import BatterySpec
    from projects.flywheel.dispatch import ThresholdController, solve_perfect_foresight

    idx = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="UTC")
    prices = pd.Series(
        70 + 50 * np.sin(2 * np.pi * np.arange(len(idx)) / 24) + RNG.normal(0, 8, len(idx)),
        index=idx,
    )
    spec = BatterySpec(power_mw=10, energy_mwh=20, degradation_cost_eur_per_mwh=1.0)

    optimum = solve_perfect_foresight(prices, spec).profit_eur
    heuristic = ThresholdController(spec=spec).run(prices).profit_eur
    assert optimum >= heuristic - 1e-6


def test_battery_env_conserves_energy_and_respects_limits():
    from lowland.config import BatterySpec
    from projects.flywheel.env import BatteryEnvConfig, BatteryTradingEnv

    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    prices = pd.Series(RNG.normal(80, 30, len(idx)), index=idx)
    spec = BatterySpec(power_mw=10, energy_mwh=20)
    env = BatteryTradingEnv(prices, BatteryEnvConfig(spec=spec, episode_hours=168))

    env.reset(start=0)
    done = False
    while not done:
        _, _, done, info = env.step(float(RNG.uniform(-1, 1)))
        assert spec.soc_min - 1e-6 <= info["soc_mwh"] <= spec.soc_max + 1e-6
        assert info["charge_mw"] <= spec.power_mw + 1e-6
        assert info["discharge_mw"] <= spec.power_mw + 1e-6
        assert not (info["charge_mw"] > 1e-6 and info["discharge_mw"] > 1e-6)


# --------------------------------------------------------------------------------------
# Scenario generation
# --------------------------------------------------------------------------------------


def test_copula_scenarios_reproduce_the_input_marginals():
    """The copula must preserve project 1's marginals while imposing dependence."""
    from projects.flywheel.scenarios import ScenarioGenerator

    H = 36
    base = 80 + 30 * np.sin(2 * np.pi * np.arange(H) / 24)
    z = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])
    q = base[:, None] + z[None, :] * 20.0

    gen = ScenarioGenerator(quantiles=QUANTILES, seed=3)
    diag = gen.diagnostics(q, n_scenarios=4000)
    # Each nominal quantile should be reproduced to within a few EUR/MWh.
    assert (diag["mean_abs_error"] < 4.0).all(), diag


def test_copula_scenarios_are_temporally_correlated():
    """Independent sampling would destroy the price path; the copula must not."""
    from projects.flywheel.scenarios import ScenarioGenerator

    H = 48
    z = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])
    q = np.full(H, 80.0)[:, None] + z[None, :] * 25.0

    paths = ScenarioGenerator(quantiles=QUANTILES, seed=5).sample(q, n_scenarios=600)
    lag1 = np.mean([np.corrcoef(p[:-1], p[1:])[0, 1] for p in paths])
    assert lag1 > 0.5, f"scenarios show lag-1 correlation of only {lag1:.2f}"


# --------------------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def long_panel() -> pd.DataFrame:
    """Three years of load, the minimum for a day-of-year climatology.

    The detector deliberately refuses to run on less than a year of *reference* history
    (excluding the recent window it is testing), because a climatology built from a
    partial year would compare, say, a December day against a June baseline and flag the
    whole winter. The shorter fixture used elsewhere cannot exercise it.
    """
    idx = pd.date_range("2022-01-01", periods=24 * 1100, freq="h", tz="UTC")
    t = np.arange(len(idx))
    seasonal = 2500 * np.cos(2 * np.pi * (t / 24 - 15) / 365.25)
    daily = 1800 * np.sin(2 * np.pi * t / 24)
    load = 12000 + seasonal + daily + RNG.normal(0, 250, len(idx))
    return pd.DataFrame({"load": load}, index=idx).rename_axis("timestamp")


def test_provisional_tail_detector_finds_an_injected_shortfall(long_panel):
    """The detector must flag a settlement-style under-reporting tail, and only the tail."""
    from lowland.dataset import detect_provisional_tail

    df = long_panel.copy()
    assert detect_provisional_tail(df) is None, "clean data should not be flagged"

    cutoff = df.index.max().normalize() - pd.Timedelta(days=12)
    df.loc[df.index >= cutoff, "load"] *= 0.72
    found = detect_provisional_tail(df)
    assert found is not None, "an injected 28% shortfall should be detected"
    assert abs((found - cutoff).days) <= 2, f"detected {found}, expected near {cutoff}"


def test_provisional_detector_ignores_an_isolated_low_day(long_panel):
    """A single low-demand day mid-history is not a settlement tail and must not be flagged.

    This is the property that separates the detector from a naive threshold: it requires
    the shortfall to run contiguously to the end of the sample, so a public holiday or an
    industrial shutdown does not silently delete a chunk of training data.
    """
    from lowland.dataset import detect_provisional_tail

    df = long_panel.copy()
    mid = df.index[len(df) // 2].normalize()
    window = (df.index >= mid) & (df.index < mid + pd.Timedelta(days=2))
    df.loc[window, "load"] *= 0.60
    assert detect_provisional_tail(df) is None
