"""Project 3 experiment runner: causal effect, regimes, generator, scenarios.

Usage
-----
    python -m projects.vantage.run
    python -m projects.vantage.run --no-generative
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from lowland.config import MODELS_DIR, REPORTS_DIR
from lowland.dataset import load_panel
from lowland.utils import get_logger, set_seed
from projects.vantage import merit_order
from projects.vantage.counterfactual import (
    SCENARIOS,
    SupplyCurveModel,
    causal_cross_check,
    simulate,
)
from projects.vantage.regimes import fit_price_regimes

log = get_logger("p3.run")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-generative", action="store_true")
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--cvae-epochs", type=int, default=400)
    args = ap.parse_args()

    set_seed()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    summary: dict[str, object] = {}

    # ---------------------------------------------------------------- causal effect ----
    log.info("=" * 70)
    log.info("1/4  merit-order effect: OLS vs IV vs DML")
    log.info("=" * 70)
    effects, context = merit_order.run_all(panel)
    effects.to_csv(REPORTS_DIR / "vantage_merit_order_effects.csv", index=False)
    print("\n===== merit-order effect (EUR/MWh per GW of renewable output) =====")
    print(effects.to_string(index=False))
    print("\ncontext:", json.dumps(context, indent=2, default=str))
    summary["merit_order"] = {"table": effects.to_dict(orient="records"), "context": context}

    causal_coef = float(
        effects.loc[effects["method"].str.startswith("IV"), "eur_per_mwh_per_gw"].iloc[0]
    )

    if not args.no_heterogeneity:
        het_year = merit_order.heterogeneous_effects(panel, by="year")
        het_peak = merit_order.heterogeneous_effects(panel, by="peak")
        het_year.to_csv(REPORTS_DIR / "vantage_merit_order_by_year.csv", index=False)
        het_peak.to_csv(REPORTS_DIR / "vantage_merit_order_by_peak.csv", index=False)
        if len(het_year):
            print("\n===== effect by year =====")
            print(het_year[["year", "eur_per_mwh_per_gw", "std_error", "first_stage_F", "n"]].to_string(index=False))
        summary["heterogeneity_year"] = het_year.to_dict(orient="records")
        summary["heterogeneity_peak"] = het_peak.to_dict(orient="records")

    # ---------------------------------------------------------------- regimes ----------
    log.info("=" * 70)
    log.info("2/4  price regimes (Gaussian HMM)")
    log.info("=" * 70)
    model, assign, extra = fit_price_regimes(panel, select_states=(2, 3, 4, 5))
    assign.to_parquet(REPORTS_DIR / "vantage_regimes_daily.parquet")
    reg_summary = model.summary()
    for col in ("level", "volatility", "spread", "neg_hours"):
        pass  # summary already rescaled inside fit_price_regimes when returned as `extra`
    extra.to_csv(REPORTS_DIR / "vantage_regime_selection.csv", index=False)

    # Re-derive a readable summary in original units.
    readable = (
        assign.groupby("state")
        .agg(
            n_days=("level", "size"),
            mean_price=("level", "mean"),
            mean_volatility=("volatility", "mean"),
            mean_spread=("spread", "mean"),
            mean_negative_hours=("neg_hours", "mean"),
            first_day=("level", lambda s: str(s.index.min().date())),
            last_day=("level", lambda s: str(s.index.max().date())),
        )
        .round(2)
    )
    readable["expected_duration_days"] = model.summary()["expected_duration_h"].to_numpy()
    readable.to_csv(REPORTS_DIR / "vantage_regimes_summary.csv")
    print("\n===== price regimes =====")
    print(readable.to_string())
    print("\nstate-count selection by BIC:")
    print(extra.to_string(index=False))
    summary["regimes"] = {
        "n_states": model.n_states,
        "summary": json.loads(readable.to_json(orient="index")),
        "transition_matrix": model.trans_.round(4).tolist(),
    }

    # ---------------------------------------------------------------- generative -------
    gen_eval = pd.DataFrame()
    if not args.no_generative:
        log.info("=" * 70)
        log.info("3/4  conditional VAE over daily price profiles")
        log.info("=" * 70)
        from projects.vantage.generative import (
            CVAEConfig,
            ProfileGenerator,
            build_daily_profiles,
            evaluate_generator,
        )

        X, C, days, meta = build_daily_profiles(panel)
        n_val = max(64, int(len(X) * 0.15))
        gen = ProfileGenerator(cfg=CVAEConfig(max_epochs=args.cvae_epochs))
        gen.fit(X, C, meta)
        # Evaluate in absolute EUR/MWh by adding the observed daily level back onto the
        # generated shape, so the spread and negative-hour statistics are directly
        # comparable with reality.
        gen_eval = evaluate_generator(
            gen, X[-n_val:], C[-n_val:], meta, n_samples=25,
            day_mean=meta["day_mean"][-n_val:],
        )
        gen_eval.to_csv(REPORTS_DIR / "vantage_generator_eval.csv", index=False)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(
            {"state_dict": gen.model.state_dict(), "meta": meta, "cfg": gen.cfg.__dict__},
            MODELS_DIR / "vantage_cvae.pt",
        )
        print("\n===== generated vs real daily price profiles (held-out days) =====")
        print(gen_eval.to_string(index=False))
        summary["generative"] = {
            "n_parameters": gen.n_parameters(),
            "n_days": meta["n_days"],
            "evaluation": gen_eval.to_dict(orient="records"),
        }

    # ---------------------------------------------------------------- scenarios --------
    log.info("=" * 70)
    log.info("4/4  counterfactual build-out scenarios")
    log.info("=" * 70)
    curve = SupplyCurveModel().fit(panel)

    results = {}
    rows = []
    for sc in SCENARIOS:
        res = simulate(panel, curve, sc)
        results[sc.name] = res
        row = {"scenario": sc.name, **res.metrics, "extrapolation_share": res.extrapolation_share}
        rows.append(row)
        res.hourly.to_parquet(REPORTS_DIR / f"vantage_scenario_{sc.name.replace(' ', '_')}.parquet")

    table = pd.DataFrame(rows).set_index("scenario")
    table.to_csv(REPORTS_DIR / "vantage_scenarios.csv")

    base = results["Base (observed)"]

    # The cross-check needs a demand-neutral twin of each scenario: the IV coefficient is
    # identified holding load constant, so comparing it against a scenario that also grows
    # demand would net a merit-order effect against a demand effect pushing the other way.
    from dataclasses import replace

    checks = {}
    for sc in SCENARIOS:
        if sc.name == "Base (observed)":
            continue
        neutral = replace(sc, demand_growth=0.0, storage_share_of_peak=0.0)
        res_neutral = simulate(panel, curve, neutral)
        checks[sc.name] = causal_cross_check(base, res_neutral, causal_coef)
        checks[sc.name]["neutral_mean_price"] = res_neutral.metrics["mean_price_eur_mwh"]
    pd.DataFrame(checks).T.to_csv(REPORTS_DIR / "vantage_causal_cross_check.csv")

    pd.set_option("display.width", 220)
    show = [
        "mean_price_eur_mwh", "negative_price_share", "mean_daily_spread_eur",
        "vre_share_of_load", "curtailment_rate", "wind_capture_rate",
        "co2_mt_per_year", "extrapolation_share",
    ]
    print("\n===== counterfactual scenarios =====")
    print(table[show].round(3).to_string())
    print("\n===== structural model vs linear causal projection =====")
    print(pd.DataFrame(checks).T.round(2).to_string())

    summary["scenarios"] = json.loads(table.to_json(orient="index"))
    summary["causal_cross_check"] = checks
    (REPORTS_DIR / "vantage_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("project 3 results written to %s", REPORTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
