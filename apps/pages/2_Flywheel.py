"""Flywheel - battery dispatch."""

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

from apps._shared import chart, label, load_csv, load_json, load_parquet, missing_artifact, relabel
from apps._theme import eyebrow, lede, note, page_setup, rule, tiles
from lowland.viz import apply_theme, line_comparison, metric_bars

th = page_setup("Flywheel")

eyebrow("02 — decide")
st.markdown("# Flywheel")
lede("What the forecast is worth to a 25 MW battery, in euros.")

table = load_csv("flywheel_controllers.csv")
if table.empty:
    missing_artifact("Results", "python -m projects.flywheel.run --model Ensemble")
    st.stop()

table = table.rename(columns={table.columns[0]: "controller"}).set_index("controller")
summary = load_json("flywheel_summary.json")
bat = summary.get("battery", {})
win = summary.get("window", {})

ranked = table.drop(index="PerfectForesight", errors="ignore").sort_values(
    "pct_of_optimum", ascending=False
)
best = ranked.index[0] if len(ranked) else None
det = table.loc["DeterministicMPC"] if "DeterministicMPC" in table.index else None
sto = table.loc["StochasticMPC"] if "StochasticMPC" in table.index else None

tiles(
    [
        {"label": "asset", "value": f"{bat.get('power_mw', 25):.0f} MW / {bat.get('energy_mwh', 50):.0f} MWh"},
        {"label": "window", "value": f"{win.get('hours', 0):,} h",
         "delta": f"{str(win.get('start',''))[:10]} → {str(win.get('end',''))[:10]}"},
        {"label": "best", "value": label(str(best)), "text_value": True,
         "delta": f"{ranked.iloc[0]['pct_of_optimum']:.1f}% of optimum" if best else "",
         "delta_state": "good"},
        {"label": "no-forecast baseline", "value": f"{table.loc['Threshold', 'pct_of_optimum']:.0f}%"
         if "Threshold" in table.index else "—", "delta": "threshold rule"},
    ]
)

# ---------------------------------------------------------------------------------------

st.markdown("## Controllers")
note(
    "Perfect foresight is a MILP on the realised prices — unachievable, but it bounds "
    "everything else, which makes results comparable across calm and volatile weeks."
)

c1, c2 = st.columns(2, gap="medium")
with c1:
    s = table["pct_of_optimum"].drop(index="PerfectForesight", errors="ignore")
    f = metric_bars(
        relabel(s), th=th, title="Share of optimum captured", subtitle="%",
        x_title="% of perfect foresight", height=320, lower_is_better=False, value_fmt="{:.1f}%",
    )
    chart(f, th, key="pct_opt")
with c2:
    f = metric_bars(
        relabel(table["profit_eur"]), th=th, title="Profit", subtitle="EUR over the window",
        x_title="EUR", height=320, lower_is_better=False, value_fmt="€{:,.0f}",
        highlight="perfect foresight",
    )
    chart(f, th, key="profit")

if det is not None and sto is not None:
    st.markdown("#### Where scenarios actually help")
    risk = pd.DataFrame(
        {
            "MPC, point forecast": [det["profit_eur"], det["cycles"], det["worst_day_eur"], det["cvar5_daily_eur"]],
            "MPC, scenarios": [sto["profit_eur"], sto["cycles"], sto["worst_day_eur"], sto["cvar5_daily_eur"]],
        },
        index=["profit €", "cycles", "worst day €", "CVaR 5% €"],
    )
    st.dataframe(risk.round(0), use_container_width=True)
    note(
        "Barely any difference in profit. The gap is in the tail — fewer cycles, and a worst "
        "day that stays positive."
    )

with st.expander("all controllers"):
    st.dataframe(relabel(table).round(2), use_container_width=True)

# ---------------------------------------------------------------------------------------

st.markdown("## Dispatch")

available = [c for c in table.index if not load_parquet(f"flywheel_schedule_{c}.parquet").empty]
pick = st.selectbox("controller", available, index=0, format_func=label)
sched = load_parquet(f"flywheel_schedule_{pick}.parquet")

if not sched.empty:
    sched.index = pd.to_datetime(sched.index, utc=True)
    days = st.slider("days", 3, min(21, max(4, len(sched) // 24)), 7)
    w = sched.tail(days * 24)

    f = go.Figure()
    f.add_trace(
        go.Scatter(x=w.index, y=w["price"], mode="lines",
                   line=dict(color=th.series(0), width=2), name="price",
                   hovertemplate="€%{y:,.1f}<extra></extra>")
    )
    f.add_trace(
        go.Scatter(x=w.index, y=np.where(w["charge_mw"] > 0, w["price"], np.nan), mode="markers",
                   marker=dict(symbol="triangle-down", color=th.series(2), size=10,
                               line=dict(width=1.5, color=th.surface)),
                   name="charge", hovertemplate="charge €%{y:,.1f}<extra></extra>")
    )
    f.add_trace(
        go.Scatter(x=w.index, y=np.where(w["discharge_mw"] > 0, w["price"], np.nan), mode="markers",
                   marker=dict(symbol="triangle-up", color=th.series(1), size=10,
                               line=dict(width=1.5, color=th.surface)),
                   name="discharge", hovertemplate="discharge €%{y:,.1f}<extra></extra>")
    )
    apply_theme(f, th, title=label(pick), subtitle="markers show the hours it acted",
                y_title="EUR/MWh", height=360)
    chart(f, th, key="dispatch")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        f2 = line_comparison(
            w[["soc_mwh"]].rename(columns={"soc_mwh": "state of charge"}), th=th, fill_first=True,
            title="State of charge", subtitle=f"MWh of {bat.get('energy_mwh', 50):.0f}",
            y_title="MWh", height=250,
        )
        chart(f2, th, key="soc")
    with c2:
        f3 = line_comparison(
            w[["profit_eur"]].cumsum().rename(columns={"profit_eur": "cumulative"}),
            th=th, fill_first=True, title="Cumulative profit", subtitle="EUR",
            y_title="EUR", height=250,
        )
        chart(f3, th, table=w[["price", "charge_mw", "discharge_mw", "soc_mwh", "profit_eur"]].round(2),
              key="cumprofit")

# ---------------------------------------------------------------------------------------

pareto = load_csv("flywheel_pareto.csv")
if len(pareto):
    st.markdown("## Profit vs congestion relief")

    f = go.Figure()
    f.add_trace(
        go.Scatter(
            x=pareto["net_grid_contribution_mwh"], y=pareto["profit_eur"],
            mode="lines+markers+text", line=dict(color=th.accent, width=2),
            marker=dict(size=11, color=th.accent, line=dict(width=2, color=th.surface)),
            text=[f"λ {v:g}" for v in pareto["congestion_lambda"]], textposition="top center",
            textfont=dict(family="JetBrains Mono, monospace", color=th.ink_muted, size=10),
            name="frontier",
            hovertemplate="λ %{text}<br>€%{y:,.0f}<br>%{x:,.0f} MWh<extra></extra>",
        )
    )
    apply_theme(
        f, th, title="Frontier", subtitle="each point is one congestion penalty λ",
        y_title="profit (EUR)", x_title="net grid contribution (MWh)",
        height=380, show_legend=False, unified_hover=False,
    )
    f.update_xaxes(showgrid=True, gridcolor=th.grid)
    chart(f, th, table=pareto.round(1), key="pareto")

    lo, hi = pareto.iloc[0], pareto.iloc[-1]
    d_mwh = hi["net_grid_contribution_mwh"] - lo["net_grid_contribution_mwh"]
    d_eur = lo["profit_eur"] - hi["profit_eur"]
    if d_mwh > 0:
        st.caption(
            f"≈ €{d_eur / d_mwh:,.0f} per MWh of congestion relief across the sweep."
        )

# ---------------------------------------------------------------------------------------

rule()

c1, c2 = st.columns(2, gap="medium")
with c1:
    diag = load_csv("flywheel_scenario_diagnostics.csv")
    if len(diag):
        with st.expander("scenario generator check"):
            st.dataframe(diag.round(2), use_container_width=True, hide_index=True)
            st.markdown(
                "Scenarios are drawn through a Gaussian copula so the marginals from "
                "Bellwether are preserved while hours stay correlated. Sampling each hour "
                "independently reproduces the marginals and destroys the path — and a "
                "battery is paid out of the path. This checks a large sample lands back on "
                "the intended quantiles."
            )
with c2:
    rl = summary.get("rl", {})
    if rl:
        with st.expander("RL agent — read with care"):
            st.dataframe(pd.DataFrame([rl]).T.rename(columns={0: "value"}), use_container_width=True)
            st.markdown(
                "The agent is **trained on surrogate forecasts** (realised prices plus an "
                "autocorrelated error calibrated to Bellwether's residuals) because genuine "
                "forecasts only exist for the 2,880-hour evaluation window — far too little "
                "for policy gradients. It is **evaluated** on the real ones. Treat its number "
                "as indicative; the MPC results don't carry that caveat."
            )

with st.expander("method notes"):
    st.markdown(
        """
The MILP keeps its on/off binaries. With strictly positive prices the relaxation is exact
and much faster, but Dutch prices are negative for several hundred hours a year — and in
those hours it genuinely pays to burn energy through the round-trip loss, so a relaxed
solver returns an infeasible schedule precisely when it matters most.

Degradation is priced (€3/MWh throughput) rather than constrained away, so the optimiser
cycles only when the spread justifies it. Profit is always evaluated against realised
prices, never the forecast.
"""
    )
