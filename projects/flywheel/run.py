"""Runner: what is a forecast actually worth to a battery?

Bellwether improves CRPS. Nobody buys CRPS. This converts forecast quality into euros
and asks three things: how much of the perfect-foresight optimum each controller captures
(the % normalisation is what makes weeks comparable -- EUR 40k in a volatile week can be
worse than EUR 15k in a calm one), whether scenarios beat the median, and what congestion
relief costs.

Caveat on the RL number, up front: the agent is *trained* on surrogate forecasts --
realised prices plus an autocorrelated error calibrated to Bellwether's out-of-sample
residuals -- because genuine forecasts only exist for the 120-day evaluation window,
nowhere near enough for policy gradients. It's *evaluated* on the real ones. The surrogate
gets the error magnitude and autocorrelation right but not every quirk of a real forecast,
so treat it as indicative. The MPC numbers don't carry that caveat.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from lowland.config import DEFAULT_BATTERY, MODELS_DIR, QUANTILES, REPORTS_DIR, BatterySpec
from lowland.dataset import congestion_threshold, load_panel
from lowland.utils import get_logger, set_seed
from projects.flywheel.dispatch import (
    ThresholdController,
    deterministic_mpc,
    solve_perfect_foresight,
    stochastic_mpc,
)
from projects.flywheel.scenarios import ScenarioGenerator, estimate_error_correlation

log = get_logger("p2.run")

QCOLS = [f"q{q:.2f}" for q in QUANTILES]


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def load_price_forecasts(model: str = "LightGBM+CQR") -> tuple[pd.DataFrame, pd.Series]:
    """Load project 1's out-of-sample price quantiles and the realised prices."""
    path = REPORTS_DIR / "bellwether_price_da_h24_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m projects.bellwether.train --target price_da` first."
        )
    df = pd.read_parquet(path)
    available = sorted(df["model"].unique())
    if model not in available:
        fallback = "LightGBM+CQR" if "LightGBM+CQR" in available else available[0]
        log.warning("model %r not in predictions (%s); using %r", model, available, fallback)
        model = fallback

    sub = df[df["model"] == model].copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True)
    sub = sub.sort_values("timestamp").drop_duplicates("timestamp")
    q = sub.set_index("timestamp")[QCOLS]
    realised = sub.set_index("timestamp")["y_true"]
    log.info("loaded %s forecasts for %s: %s .. %s", len(q), model, q.index.min(), q.index.max())
    return q, realised


def congestion_signal(index: pd.DatetimeIndex) -> pd.Series:
    """Normalised system-stress weight in ``[0, 1]`` from realised residual load.

    Defined as how far residual load sits between its median and the P90 stress threshold.
    Charging when this is high is what a congested grid cannot absorb, so it is the term
    the co-optimisation penalises.
    """
    panel = load_panel()
    rl = panel["residual_load"].reindex(index)
    thr = congestion_threshold(panel)
    med = float(panel["residual_load"].median())
    w = ((rl - med) / max(thr - med, 1.0)).clip(0.0, 1.0)
    return w.fillna(0.0)


# --------------------------------------------------------------------------------------
# Surrogate forecasts for RL training
# --------------------------------------------------------------------------------------


def surrogate_forecast_quantiles(
    prices: pd.Series,
    residual_std: float,
    rho: float,
    quantiles: tuple[float, ...] = QUANTILES,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Build plausible forecast bands around a realised price path.

    An AR(1) error with the supplied standard deviation and lag-1 autocorrelation is added
    to the realised price to form the pseudo-median, and the band is the Gaussian quantile
    spread around it. The point is not realism for its own sake but to give the RL agent an
    observation with the *same signal-to-noise ratio* it will face at evaluation time -- an
    agent trained on perfect foresight learns a policy that collapses the moment the
    forecast is imperfect.
    """
    rng = np.random.default_rng(seed)
    p = prices.to_numpy(dtype=float)
    n = len(p)

    e = np.zeros(n)
    innov_sd = residual_std * np.sqrt(max(1 - rho**2, 1e-6))
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, innov_sd)

    med = p + e
    from scipy import stats as sstats

    z = sstats.norm.ppf(np.asarray(quantiles))
    band = med[:, None] + z[None, :] * residual_std
    return np.sort(band, axis=1)


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------


def run_controllers(
    q_frame: pd.DataFrame,
    realised: pd.Series,
    spec: BatterySpec,
    *,
    congestion_lambda: float = 0.0,
    n_scenarios: int = 10,
    horizon: int = 36,
    commit: int = 24,
    rho: float = 0.82,
    seasonal_weight: float = 0.25,
) -> dict[str, object]:
    """Run every non-RL controller on one evaluation window."""
    cw = congestion_signal(realised.index)
    med_col = "q0.50"
    results: dict[str, object] = {}

    log.info("perfect-foresight MILP over %s hours ...", len(realised))
    pf = solve_perfect_foresight(
        realised, spec, congestion_weight=cw, congestion_lambda=congestion_lambda
    )
    results["PerfectForesight"] = pf

    log.info("threshold heuristic ...")
    results["Threshold"] = ThresholdController(spec=spec).run(realised)

    log.info("deterministic MPC ...")
    results["DeterministicMPC"] = deterministic_mpc(
        realised, q_frame[med_col], spec,
        horizon=horizon, commit=commit,
        congestion_weight=cw, congestion_lambda=congestion_lambda,
    )

    log.info("stochastic MPC with %s scenarios ...", n_scenarios)
    gen = ScenarioGenerator(quantiles=QUANTILES, rho=rho, seasonal_weight=seasonal_weight)
    arr = q_frame.to_numpy(dtype=float)

    def scenario_fn(start: int, end: int) -> np.ndarray:
        return gen.sample(arr[start:end], n_scenarios=n_scenarios)

    results["StochasticMPC"] = stochastic_mpc(
        realised, scenario_fn, spec,
        horizon=horizon, commit=commit, n_scenarios=n_scenarios,
        congestion_weight=cw, congestion_lambda=congestion_lambda,
    )
    return results


def summarise(
    results: dict[str, object], spec: BatterySpec, cw: pd.Series | None = None
) -> pd.DataFrame:
    """Build the comparison table, normalised against perfect foresight."""
    pf = results.get("PerfectForesight")
    ceiling = pf.profit_eur if pf is not None else float("nan")

    rows = []
    for name, res in results.items():
        sched = res.schedule
        daily = sched["profit_eur"].resample("D").sum()
        # Grid impact is measured against the realised stress signal, not against the
        # penalty column: with the congestion weight switched off that column is
        # identically zero, so deriving grid impact from it reports zero for every
        # controller regardless of what they actually did.
        w = cw.reindex(sched.index).fillna(0.0) if cw is not None else pd.Series(0.0, index=sched.index)
        stress_charge = float((sched["charge_mw"] * w).sum())
        stress_discharge = float((sched["discharge_mw"] * w).sum())
        rows.append(
            {
                "controller": name,
                "profit_eur": res.profit_eur,
                "pct_of_optimum": 100.0 * res.profit_eur / ceiling if ceiling else np.nan,
                "eur_per_mw_year": res.profit_eur
                / spec.power_mw
                * (8760.0 / max(len(sched), 1)),
                "cycles": res.equivalent_full_cycles,
                "energy_discharged_mwh": res.energy_discharged_mwh,
                # Downside risk of the daily P&L; a controller that earns well on average
                # but loses badly on its worst days is not financeable.
                "worst_day_eur": float(daily.min()) if len(daily) else np.nan,
                "cvar5_daily_eur": float(daily[daily <= daily.quantile(0.05)].mean())
                if len(daily) > 20 else np.nan,
                "stress_charged_mwh": stress_charge,
                "stress_relieved_mwh": stress_discharge,
                "net_grid_contribution_mwh": stress_discharge - stress_charge,
                "solve_seconds": getattr(res, "solve_seconds", np.nan),
            }
        )
    return pd.DataFrame(rows).set_index("controller").round(2)


def pareto_sweep(
    q_frame: pd.DataFrame,
    realised: pd.Series,
    spec: BatterySpec,
    lambdas: list[float],
    *,
    n_scenarios: int = 8,
) -> pd.DataFrame:
    """Trace the profit-versus-congestion frontier for the stochastic controller."""
    cw = congestion_signal(realised.index)
    gen = ScenarioGenerator(quantiles=QUANTILES)
    arr = q_frame.to_numpy(dtype=float)

    def scenario_fn(start: int, end: int) -> np.ndarray:
        return gen.sample(arr[start:end], n_scenarios=n_scenarios)

    rows = []
    for lam in lambdas:
        res = stochastic_mpc(
            realised, scenario_fn, spec,
            horizon=36, commit=24, n_scenarios=n_scenarios,
            congestion_weight=cw, congestion_lambda=lam,
        )
        sched = res.schedule
        # Grid stress caused = energy charged, weighted by how stressed the system was.
        stress = float((sched["charge_mw"] * cw.reindex(sched.index).fillna(0)).sum())
        relief = float((sched["discharge_mw"] * cw.reindex(sched.index).fillna(0)).sum())
        rows.append(
            {
                "congestion_lambda": lam,
                "profit_eur": res.profit_eur,
                "stress_charged_mwh": stress,
                "stress_relieved_mwh": relief,
                "net_grid_contribution_mwh": relief - stress,
                "cycles": res.equivalent_full_cycles,
            }
        )
        log.info(
            "lambda=%6.1f  profit=%10.0f  net grid contribution=%8.1f MWh",
            lam, res.profit_eur, relief - stress,
        )
    return pd.DataFrame(rows)


def train_and_eval_rl(
    q_frame: pd.DataFrame,
    realised: pd.Series,
    spec: BatterySpec,
    *,
    n_updates: int = 80,
    train_years: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """Train PPO on surrogate forecasts, then evaluate it on the genuine ones."""
    from projects.flywheel.env import BatteryEnvConfig, BatteryTradingEnv
    from projects.flywheel.ppo import PPOAgent, PPOConfig

    panel = load_panel()
    hist = panel.loc[panel["is_complete"] == 1, "price_da"].dropna()
    hist = hist[hist.index < realised.index.min()]
    hist = hist[hist.index > hist.index.max() - pd.Timedelta(days=365 * train_years)]

    # Calibrate the surrogate's noise to project 1's realised out-of-sample error.
    err = (realised - q_frame["q0.50"]).dropna()
    resid_sd = float(err.std())
    rho = float(np.corrcoef(err.to_numpy()[1:], err.to_numpy()[:-1])[0, 1])
    rho = float(np.clip(rho, 0.1, 0.95))
    log.info("surrogate forecast noise: sd=%.2f EUR/MWh  rho=%.3f  (n=%s)", resid_sd, rho, len(err))

    train_fq = surrogate_forecast_quantiles(hist, resid_sd, rho)
    cfg = BatteryEnvConfig(spec=spec)
    train_env = BatteryTradingEnv(hist, cfg, forecast_quantiles=train_fq)

    agent = PPOAgent(obs_dim=train_env.obs_dim, cfg=PPOConfig(n_updates=n_updates))
    log.info("training PPO: %s parameters, %s training hours", f"{agent.n_parameters():,}", len(hist))
    agent.train(train_env)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(MODELS_DIR / "flywheel_ppo_battery.pt")

    eval_env = BatteryTradingEnv(
        realised, BatteryEnvConfig(spec=spec), forecast_quantiles=q_frame.to_numpy(dtype=float)
    )
    sched = eval_env.rollout(agent.policy(deterministic=True), start=0, n_hours=len(realised) - 1)

    info = {
        "n_parameters": agent.n_parameters(),
        "n_updates": n_updates,
        "train_hours": int(len(hist)),
        "surrogate_sd_eur": round(resid_sd, 2),
        "surrogate_rho": round(rho, 3),
        "final_ep_return": agent.history_[-1]["ep_return"] if agent.history_ else None,
        "device": agent.device,
    }
    return sched, info


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LightGBM+CQR", help="project 1 model to source forecasts from")
    ap.add_argument("--power-mw", type=float, default=25.0)
    ap.add_argument("--energy-mwh", type=float, default=50.0)
    ap.add_argument("--days", type=int, default=0, help="limit the evaluation window (0 = all)")
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--no-rl", action="store_true")
    ap.add_argument("--rl-updates", type=int, default=80)
    ap.add_argument("--no-pareto", action="store_true")
    args = ap.parse_args()

    set_seed()
    spec = BatterySpec(power_mw=args.power_mw, energy_mwh=args.energy_mwh)

    q_frame, realised = load_price_forecasts(args.model)
    # Keep only a contiguous, regularly spaced block: the MILP indexes by position, so a
    # gap between folds would silently shift the state-of-charge dynamics.
    q_frame, realised = _largest_contiguous_block(q_frame, realised)
    if args.days:
        keep = realised.index >= realised.index.max() - pd.Timedelta(days=args.days)
        q_frame, realised = q_frame[keep], realised[keep]
    log.info("evaluation window: %s hours, %s .. %s",
             len(realised), realised.index.min(), realised.index.max())

    preds_path = REPORTS_DIR / "bellwether_price_da_h24_predictions.parquet"
    rho, seasonal = estimate_error_correlation(
        pd.read_parquet(preds_path).query("model == @args.model"), horizon=36
    )

    results = run_controllers(
        q_frame, realised, spec, n_scenarios=args.scenarios, rho=rho, seasonal_weight=seasonal
    )

    rl_info: dict = {}
    if not args.no_rl:
        from projects.flywheel.dispatch import DispatchResult

        sched, rl_info = train_and_eval_rl(q_frame, realised, spec, n_updates=args.rl_updates)
        discharged = float(sched["discharge_mw"].sum())
        results["PPO"] = DispatchResult(
            schedule=sched,
            profit_eur=float(sched["profit_eur"].sum()),
            energy_discharged_mwh=discharged,
            equivalent_full_cycles=discharged / spec.energy_mwh,
            congestion_penalty_eur=float(sched.get("congestion_eur", pd.Series([0])).sum()),
            solver_status="ppo",
        )

    table = summarise(results, spec, cw=congestion_signal(realised.index))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "flywheel_controllers.csv")

    for name, res in results.items():
        res.schedule.to_parquet(REPORTS_DIR / f"flywheel_schedule_{name}.parquet")

    pareto = pd.DataFrame()
    if not args.no_pareto:
        pareto = pareto_sweep(q_frame, realised, spec, [0.0, 10.0, 25.0, 50.0, 100.0, 200.0])
        pareto.to_csv(REPORTS_DIR / "flywheel_pareto.csv", index=False)

    # Scenario-generator self-check, so the copula is verified rather than assumed.
    gen = ScenarioGenerator(quantiles=QUANTILES, rho=rho, seasonal_weight=seasonal)
    diag = gen.diagnostics(q_frame.to_numpy(dtype=float)[:36], n_scenarios=400)
    diag.to_csv(REPORTS_DIR / "flywheel_scenario_diagnostics.csv", index=False)

    summary = {
        "model_source": args.model,
        "battery": {"power_mw": spec.power_mw, "energy_mwh": spec.energy_mwh,
                    "round_trip_efficiency": round(spec.round_trip_efficiency, 4)},
        "window": {"hours": int(len(realised)), "start": str(realised.index.min()),
                   "end": str(realised.index.max())},
        "controllers": json.loads(table.to_json(orient="index")),
        "rl": rl_info,
        "scenario_correlation": {"rho": rho, "seasonal_weight": seasonal},
        "scenario_marginal_check_mae": float(diag["mean_abs_error"].mean()),
        "mean_daily_spread_eur": diag.attrs.get("mean_daily_spread"),
    }
    (REPORTS_DIR / "flywheel_summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")

    pd.set_option("display.width", 200)
    print("\n===== controller comparison =====")
    print(table.to_string())
    if len(pareto):
        print("\n===== congestion Pareto frontier =====")
        print(pareto.round(1).to_string(index=False))
    print("\n===== scenario marginal reproduction =====")
    print(diag.round(2).to_string(index=False))
    return 0


def _largest_contiguous_block(
    q_frame: pd.DataFrame, realised: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the longest run of consecutive hourly observations.

    Project 1's pooled predictions come from several disjoint folds. Concatenating them
    would let the optimiser carry state of charge across a multi-month gap, which is
    physically meaningless.
    """
    idx = realised.index
    gaps = idx.to_series().diff() != pd.Timedelta(hours=1)
    block_id = gaps.cumsum()
    sizes = block_id.value_counts()
    best = sizes.idxmax()
    keep = (block_id == best).to_numpy()
    log.info("using the largest contiguous block: %s of %s hours", keep.sum(), len(idx))
    return q_frame[keep], realised[keep]


if __name__ == "__main__":
    sys.exit(main())
