"""Vantage - causal effect, regimes, scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps._shared import chart, load_csv, load_json, load_parquet, missing_artifact
from apps._theme import eyebrow, lede, note, page_setup, rule, tiles
from lowland.viz import apply_theme, dot_interval, line_comparison, metric_bars

th = page_setup("Vantage")

eyebrow("03 — explain")
st.markdown("# Vantage")
lede("Why Dutch prices move, and where build-out takes them.")

effects = load_csv("vantage_merit_order_effects.csv")
if effects.empty:
    missing_artifact("Results", "python -m projects.vantage.run")
    st.stop()

summary = load_json("vantage_summary.json")
ctx = summary.get("merit_order", {}).get("context", {})


def _val(prefix: str, col: str = "eur_per_mwh_per_gw") -> float:
    row = effects[effects["method"].str.startswith(prefix)]
    return float(row[col].iloc[0]) if len(row) else np.nan


iv, ols, dml = _val("IV"), _val("OLS"), _val("DML")
fs_f = _val("IV", "first_stage_F")

st.markdown("## Merit-order effect")
note("How much an extra GW of wind and solar output moves the day-ahead price.")

tiles(
    [
        {"label": "IV estimate", "value": f"−€{abs(iv):,.2f}", "delta": "per MWh, per GW",
         "delta_state": "good"},
        {"label": "naive OLS", "value": f"−€{abs(ols):,.2f}",
         "delta": f"{ols - iv:+.2f} bias" if np.isfinite(ols) and np.isfinite(iv) else None},
        {"label": "double ML", "value": f"−€{abs(dml):,.2f}", "delta": "cross-fitted"},
        {"label": "first-stage F", "value": f"{fs_f:,.0f}" if np.isfinite(fs_f) else "—",
         "delta": "weak-instrument bar ≈ 10", "delta_state": "good"},
    ]
)

c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    f = dot_interval(
        labels=[m.split(" (")[0] for m in effects["method"]],
        centres=effects["eur_per_mwh_per_gw"].tolist(),
        lows=effects["ci_low"].tolist(),
        highs=effects["ci_high"].tolist(),
        th=th, title="Estimates with 95% intervals",
        subtitle="EUR/MWh per GW · Newey–West errors",
        x_title="EUR/MWh per GW", height=270,
    )
    chart(f, th, key="effects")
    st.dataframe(
        effects[["method", "eur_per_mwh_per_gw", "std_error", "ci_low", "ci_high", "n"]].round(3),
        use_container_width=True, hide_index=True,
    )
with c2:
    st.markdown("#### Why not just regress price on wind?")
    st.markdown(
        """
Three problems, pulling in different directions:

- cold still evenings have high demand **and** low wind, so the naive fit charges wind with
  a price effect that belongs to demand;
- when prices go deeply negative, wind farms curtail — low price causes low output;
- the feed misses most behind-the-meter solar, and measurement error attenuates.

**Instrument:** hub-height wind speed and irradiance where Dutch capacity actually sits.
Price doesn't cause wind. Temperature is a *control*, not an instrument, since it reaches
price through demand too.
"""
    )
    if ctx:
        st.caption(
            f"{ctx.get('n_hours', 0):,} hours, {str(ctx.get('sample_start'))[:10]} → "
            f"{str(ctx.get('sample_end'))[:10]}."
        )

het = load_csv("vantage_merit_order_by_year.csv")
if len(het):
    f = line_comparison(
        het.set_index("year")["eur_per_mwh_per_gw"].to_frame("effect"),
        th=th, title="By year", subtitle="re-estimated by IV within each year",
        y_title="EUR/MWh per GW", x_title="year", height=290,
    )
    chart(f, th, table=het.round(3), key="het")
    note(
        "The coefficient is the slope of the residual supply curve — it steepens when gas is "
        "expensive and flattens as the system normalises. Holding it fixed in a 2030 model is "
        "a mistake."
    )

rule()

# ---------------------------------------------------------------------------------------

st.markdown("## Price regimes")
note("Gaussian HMM on daily level, volatility, spread and negative-hour count. No labels, no dates.")

reg = load_csv("vantage_regimes_summary.csv")
daily = load_parquet("vantage_regimes_daily.parquet")

c1, c2 = st.columns([2, 3], gap="medium")
with c1:
    if len(reg):
        st.dataframe(reg.round(2), use_container_width=True, hide_index=True)
    sel = load_csv("vantage_regime_selection.csv")
    if len(sel) and "bic" in sel.columns:
        with st.expander("state count (BIC)"):
            st.dataframe(sel.round(1), use_container_width=True, hide_index=True)
with c2:
    if len(daily):
        daily.index = pd.to_datetime(daily.index, utc=True)
        f = go.Figure()
        for k in sorted(daily["state"].unique()):
            m = daily["state"] == k
            f.add_trace(
                go.Scatter(
                    x=daily.index[m], y=daily["level"][m], mode="markers",
                    marker=dict(size=4.5, color=th.series(int(k) % 7), opacity=0.85),
                    name=f"regime {k}",
                    hovertemplate=f"regime {k}<br>%{{x|%b %Y}}  €%{{y:.0f}}<extra></extra>",
                )
            )
        apply_theme(f, th, title="Daily mean price by inferred regime",
                    subtitle="Viterbi decoding", y_title="EUR/MWh", height=370,
                    unified_hover=False)
        chart(f, th, table=daily.tail(200).round(2), key="regimes")

rule()

# ---------------------------------------------------------------------------------------

st.markdown("## Build-out scenarios")

scen = load_csv("vantage_scenarios.csv")
if len(scen):
    scen = scen.rename(columns={scen.columns[0]: "scenario"}).set_index("scenario")
    show = [c for c in ("mean_price_eur_mwh", "negative_price_share", "mean_daily_spread_eur",
                        "vre_share_of_load", "curtailment_rate", "wind_capture_rate",
                        "co2_mt_per_year", "extrapolation_share") if c in scen.columns]
    disp = scen[show].copy()
    disp.columns = ["price €/MWh", "negative share", "spread €", "VRE share",
                    "curtailment", "capture rate", "CO₂ Mt/yr", "extrapolation"][: len(show)]
    st.dataframe(disp.round(3), use_container_width=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        f = metric_bars(
            scen["mean_price_eur_mwh"], th=th, title="Mean price", subtitle="EUR/MWh",
            x_title="EUR/MWh", height=290, value_fmt="€{:,.1f}",
        )
        chart(f, th, key="scen_price")
    with c2:
        if "wind_capture_rate" in scen.columns:
            f = metric_bars(
                scen["wind_capture_rate"], th=th, title="Wind capture rate",
                subtitle="wind-weighted price ÷ average price", x_title="ratio",
                height=290, lower_is_better=False, value_fmt="{:.3f}",
            )
            chart(f, th, key="scen_capture")
            note(
                "Capture rate, not average price, decides whether an unsubsidised project is "
                "financeable — output concentrates in the hours its own abundance makes cheap."
            )

    cross = load_csv("vantage_causal_cross_check.csv")
    if len(cross):
        with st.expander("structural model vs the causal estimate"):
            cross = cross.rename(columns={cross.columns[0]: "scenario"})
            st.dataframe(cross.round(2), use_container_width=True, hide_index=True)
            st.markdown(
                "The linear column multiplies the IV coefficient by the change in renewable "
                "output with demand held fixed — the comparison the coefficient is identified "
                "for. The agreement ratio falls as scenarios grow (0.77 → 0.70 → 0.68), which "
                "is the convexity of the merit order: extrapolating a marginal effect linearly "
                "overstates it once the system moves onto a flatter part of the curve. "
                "`extrapolation` reports how often a scenario leaves the residual-load range "
                "ever observed."
            )

# ---------------------------------------------------------------------------------------

gen = load_csv("vantage_generator_eval.csv")
if len(gen):
    rule()
    st.markdown("## Daily price shapes")
    note(
        "Conditional VAE over 24-hour profiles. The supply curve sets the level; this models "
        "the shape around it, which is what storage is paid for."
    )
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.dataframe(gen, use_container_width=True, hide_index=True)
    with c2:
        spread = gen[gen["statistic"] == "mean_daily_spread"]
        spread_pct = (
            100 * float(spread["generated"].iloc[0]) / float(spread["real"].iloc[0])
            if len(spread) else float("nan")
        )
        st.markdown("#### This one only half works")
        st.markdown(
            f"""
Median and p95 land within a few percent. But daily spread reaches only **{spread_pct:.0f}%**
of the real one and negative hours are badly under-produced — the decoder pulls toward the
conditional mean, which is what 4,241 training days over a 24-dimensional profile buys you.

Ex-post sampling from the aggregate posterior helped marginally. A heavier-tailed
likelihood or a diffusion model is the obvious next step. Leaving it in because the failure
is informative.
"""
        )
