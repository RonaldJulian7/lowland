"""Lowland - overview.

Run with:  streamlit run apps/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from apps._shared import chart, load_panel_cached
from apps._theme import card, eyebrow, lede, note, page_setup, rule, tiles
from lowland.viz import line_comparison, stacked_area

th = page_setup("Home")

eyebrow("dutch power system · 2015—2026")
st.markdown("# Lowland")
lede(
    "Three studies of the Dutch electricity system on eleven years of hourly open data. "
    "They chain together: forecast, decide, explain."
)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown(
        card("01 — forecast", "Bellwether",
             "Residual load and day-ahead price 24 h out. Quantile GBM against a "
             "sequence model, conformal calibration, and congestion risk."),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        card("02 — decide", "Flywheel",
             "What that forecast is worth to a grid battery. MILP ceiling, MPC on point "
             "and scenario forecasts, an RL agent, and a congestion trade-off."),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        card("03 — explain", "Vantage",
             "The merit-order effect with weather instruments, price regimes from an HMM, "
             "and 2030/2035 build-out scenarios."),
        unsafe_allow_html=True,
    )

rule()

# ---------------------------------------------------------------------------------------

panel = load_panel_cached()
obs = panel[panel["is_complete"] == 1]

st.markdown("## The data")

tiles(
    [
        {"label": "hourly rows", "value": f"{len(obs):,}"},
        {"label": "span", "value": f"{obs.index.min():%Y}—{obs.index.max():%Y}"},
        {"label": "mean price", "value": f"€{obs['price_da'].mean():,.0f}", "delta": "per MWh"},
        {"label": "negative-price hours", "value": f"{(obs['price_da'] < 0).sum():,}",
         "delta": f"{100 * (obs['price_da'] < 0).mean():.1f}% of hours"},
    ]
)

st.markdown("")

yearly = (
    obs.assign(year=obs.index.year)
    .groupby("year")
    .agg(mean_price=("price_da", "mean"), vre_share=("vre_share", "mean"))
)

left, right = st.columns(2, gap="medium")
with left:
    f = line_comparison(
        yearly[["mean_price"]].rename(columns={"mean_price": "mean price"}),
        th=th, fill_first=True, title="Day-ahead price", subtitle="annual mean · EUR/MWh",
        y_title="EUR/MWh", height=310,
    )
    chart(f, th, table=yearly[["mean_price"]].round(1), key="price_yr")
with right:
    f = line_comparison(
        yearly[["vre_share"]].rename(columns={"vre_share": "wind + solar"}),
        th=th, fill_first=True, title="Renewable share of load", subtitle="metered wind and solar",
        y_title="share", height=310,
    )
    chart(f, th, table=yearly[["vre_share"]].round(3), key="vre_yr")

note(
    "Two different worlds in one sample — which is why the forecasting work leans on "
    "conformal calibration and the causal work on regime modelling."
)

# ---------------------------------------------------------------------------------------

st.markdown("## Generation mix")

recent = obs.tail(24 * 21)
mix_cols = {
    "wind offshore": "wind_offshore",
    "wind onshore": "wind_onshore",
    "solar": "solar",
    "nuclear": "nuclear",
    "fossil gas": "fossil_gas",
    "fossil coal": "fossil_hard_coal",
}
mix = pd.DataFrame({k: recent[v] for k, v in mix_cols.items() if v in recent.columns})
other_cols = [c for c in ("biomass", "waste", "others", "fossil_oil") if c in recent.columns]
if other_cols:
    mix["other"] = recent[other_cols].sum(axis=1)

f = stacked_area(
    mix, th=th, title="Hourly generation", subtitle="MW · last three settled weeks",
    y_title="MW", height=390,
)
chart(f, th, table=mix.resample("D").mean().round(0), key="mix")

