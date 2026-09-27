"""Bellwether - forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from apps._shared import (
    QCOLS,
    chart,
    label,
    load_csv,
    load_json,
    load_panel_cached,
    missing_artifact,
    predictions_frame,
    relabel,
)
from apps._theme import eyebrow, lede, note, page_setup, rule, tiles
from lowland.config import QUANTILES
from lowland.dataset import congestion_threshold
from lowland.viz import fan_chart, line_comparison, metric_bars, reliability_diagram
from projects.bellwether.congestion import build_congestion_report, risk_calibration

th = page_setup("Bellwether")

eyebrow("01 — forecast")
st.markdown("# Bellwether")
lede("Residual load and day-ahead price, 24 hours out, with intervals that hold up.")

target = st.sidebar.selectbox(
    "target", ["residual_load", "price_da"],
    format_func=lambda t: "residual load" if t == "residual_load" else "day-ahead price",
)
unit = "MW" if target == "residual_load" else "EUR/MWh"

# The splitter always takes the most recent windows, so the default backtest sits in
# summer. Truncating the panel at 1 March re-runs the same protocol on Nov-Feb instead.
# Congestion is a winter problem, so testing only on summer would have left a hole where
# the important case should be.
season = st.sidebar.radio(
    "backtest window", ["summer", "winter"],
    format_func=lambda s: "May – Aug 2026" if s == "summer" else "Nov 2025 – Feb 2026",
)
sfx = "" if season == "summer" else "_winter"

metrics = load_csv(f"bellwether_{target}_h24{sfx}_metrics.csv")
if metrics.empty:
    missing_artifact(
        "Results",
        f"python -m projects.bellwether.train --target {target} --folds 4"
        + (" --until 2026-03-01 --label winter" if sfx else ""),
    )
    st.stop()

metrics = metrics.rename(columns={metrics.columns[0]: "model"}).set_index("model")
summary = load_json(f"bellwether_{target}_h24{sfx}_summary.json")
preds = predictions_frame(target, suffix=sfx)
models = list(metrics.index)

best = metrics["crps"].idxmin()
naive = "SeasonalNaive" if "SeasonalNaive" in metrics.index else models[0]
skill = 100 * (1 - metrics.loc[best, "crps"] / metrics.loc[naive, "crps"])
picp = metrics.loc[best, "picp90"] if "picp90" in metrics.columns else np.nan

tiles(
    [
        {"label": "best model", "value": label(best), "text_value": True},
        {"label": f"MAE · {unit}", "value": f"{metrics.loc[best, 'mae']:,.1f}"},
        {"label": "CRPS vs seasonal naive", "value": f"−{skill:.0f}%", "delta_state": "good"},
        {"label": "90% coverage", "value": f"{picp:.1%}" if np.isfinite(picp) else "—",
         "delta": "target 90%"},
    ]
)

# ---------------------------------------------------------------------------------------

st.markdown("## Model comparison")

c1, c2 = st.columns(2, gap="medium")
with c1:
    f = metric_bars(
        relabel(metrics["crps"]), th=th, title="CRPS", subtitle=f"lower is better · {unit}",
        x_title=unit, height=350,
    )
    chart(f, th, key="crps_bars")
with c2:
    metric = st.selectbox("metric", ["mae", "rmse", "pinball", "winkler90", "mpiw90"], index=0)
    f = metric_bars(
        relabel(metrics[metric]), th=th, title=metric.upper(), subtitle=f"lower is better · {unit}",
        x_title=unit, height=350,
    )
    chart(f, th, key="metric_bars")

with st.expander("all metrics"):
    st.dataframe(relabel(metrics).round(3), use_container_width=True)

# ---------------------------------------------------------------------------------------

st.markdown("## Calibration")
note(
    "Raw quantile models undercover out of sample. Conformal correction is fitted on a "
    "held-out block and needs no distributional assumption."
)
if sfx and {"LightGBM+CQR", "LightGBM+Mondrian"} <= set(metrics.index):
    cqr_c = metrics.loc["LightGBM+CQR", "picp90"]
    mon_c = metrics.loc["LightGBM+Mondrian", "picp90"]
    raw_c = metrics.loc["LightGBM", "picp90"] if "LightGBM" in metrics.index else float("nan")
    if mon_c > cqr_c:
        note(
            f"Winter is where the group-conditional version earns its keep. Raw coverage "
            f"drops to <b>{raw_c:.3f}</b> here — worse than summer — and plain conformal "
            f"recovers it to {cqr_c:.3f}, but calibrating within hour-block and season "
            f"gets to <b>{mon_c:.3f}</b>. In summer the two were indistinguishable."
        )

pick = st.multiselect(
    "models", models,
    default=[m for m in ("LightGBM", "LightGBM+CQR", "Ensemble") if m in models] or models[:2],
    format_func=label,
)
curves = {}
for m in pick:
    c = load_csv(f"bellwether_{target}_h24{sfx}_reliability_{m.replace('+', '_')}.csv")
    if len(c):
        curves[label(m)] = c

c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    if curves:
        f = reliability_diagram(curves, th=th, height=400)
        chart(f, th, table=pd.concat([v.assign(model=k) for k, v in curves.items()]).round(3),
              key="reliability")
with c2:
    cov_cols = [c for c in ("picp90", "picp80", "picp50") if c in metrics.columns]
    if cov_cols:
        cov = metrics.loc[pick, cov_cols] if pick else metrics[cov_cols]
        disp = relabel(cov)
        disp.columns = ["90%", "80%", "50%"][: len(cov_cols)]
        nominal = pd.Series([0.90, 0.80, 0.50][: len(cov_cols)], index=disp.columns, name="target")
        st.markdown("#### Empirical coverage")
        st.dataframe(pd.concat([disp, nominal.to_frame().T]).round(3), use_container_width=True)

# ---------------------------------------------------------------------------------------

st.markdown("## Forecast")

if not preds.empty:
    avail = [m for m in models if m in set(preds["model"])]
    model_for_fan = st.selectbox(
        "model", avail, index=avail.index(best) if best in avail else 0,
        key="fan_model", format_func=label,
    )
    sub = preds[preds["model"] == model_for_fan].set_index("timestamp").sort_index()
    block = (sub.index.to_series().diff() != pd.Timedelta(hours=1)).cumsum()
    win = sub[block == block.value_counts().index[0]]
    n_days = st.slider("days", 3, min(30, max(4, len(win) // 24)), min(10, max(4, len(win) // 24)))
    win = win.tail(n_days * 24)

    f = fan_chart(
        win.index, win[QCOLS].to_numpy(), QUANTILES, observed=win["y_true"], th=th,
        title=label(model_for_fan),
        subtitle=f"24 h ahead · 50 / 80 / 90% intervals · {unit}",
        y_title=unit, height=450,
    )
    tbl = win[["y_true", "q0.50", "q0.05", "q0.95"]].round(1)
    tbl.columns = ["observed", "median", "p5", "p95"]
    chart(f, th, table=tbl, key="fan")

# ---------------------------------------------------------------------------------------

if target == "residual_load" and not preds.empty:
    st.markdown("## Congestion risk")

    panel = load_panel_cached()
    historic_thr = congestion_threshold(panel)
    model_for_risk = best if best in set(preds["model"]) else list(set(preds["model"]))[0]
    sub = preds[preds["model"] == model_for_risk].set_index("timestamp").sort_index()
    q = sub[QCOLS].to_numpy()
    window_thr = float(np.nanquantile(sub["y_true"].to_numpy(), 0.90))

    thr = st.slider(
        "threshold · MW",
        float(min(window_thr, historic_thr) * 0.7),
        float(max(window_thr, historic_thr) * 1.15),
        float(window_thr), step=100.0,
    )
    note(
        f"Defaults to the P90 of this window ({window_thr:,.0f} MW). Full-history P90 is "
        f"{historic_thr:,.0f} MW — the backtest falls in summer, so annual peaks don't bind."
    )

    report = build_congestion_report(sub.index, q, QUANTILES, thr, ramp_threshold_mw=2500.0)
    rc = risk_calibration(sub["y_true"].to_numpy(), q, QUANTILES, thr)

    tiles(
        [
            {"label": "hours over 50% risk", "value": f"{report.hours_above_50pct:,}"},
            {"label": "peak expected exceedance", "value": f"{report.max_expected_exceedance_mw:,.0f} MW"},
            {"label": "Brier score", "value": f"{rc.attrs.get('brier', float('nan')):.4f}"},
            {"label": "Brier skill", "value": f"{rc.attrs.get('brier_skill', float('nan')):.1%}",
             "delta": "vs climatology", "delta_state": "good"},
        ]
    )

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        risk = report.table[["p_exceed"]].tail(24 * 14)
        risk.columns = ["P(exceed)"]
        f = line_comparison(
            risk, th=th, fill_first=True, title="Exceedance probability",
            subtitle=f"threshold {thr:,.0f} MW · last 14 days",
            y_title="probability", height=330,
        )
        chart(f, th, table=report.table.tail(24 * 14).round(3), key="risk_line")
    with c2:
        st.markdown("#### Risk calibration")
        if len(rc):
            st.dataframe(rc.round(3), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------------------

st.markdown("## Drivers")

c1, c2 = st.columns(2, gap="medium")
with c1:
    imp = load_csv(f"bellwether_{target}_h24{sfx}_importance.csv")
    if len(imp):
        imp.columns = ["group", "importance"]
        f = metric_bars(
            imp.set_index("group")["importance"], th=th, title="Feature groups",
            subtitle="share of split gain", x_title="share", height=300,
            lower_is_better=False, value_fmt="{:.1%}",
        )
        chart(f, th, table=imp.round(4), key="importance")
with c2:
    dm = load_csv(f"bellwether_{target}_h24{sfx}_dm.csv")
    if len(dm):
        st.markdown("#### Diebold–Mariano vs seasonal naive")
        show = dm[["model", "dm_stat", "p_value"]].copy()
        show["model"] = show["model"].map(label)
        show["sig"] = np.where(show["p_value"] < 0.05, "yes", "no")
        st.dataframe(show.round(4), use_container_width=True, hide_index=True)
        h = summary.get("dm_head_to_head") or {}
        if h:
            p = h.get("p_value", float("nan"))
            st.caption(
                f"{label(h.get('best',''))} vs {label(h.get('runner_up',''))}: "
                f"DM {h.get('dm_stat', float('nan')):.2f}, p={p:.3f}"
                + ("" if p < 0.05 else " — not significant.")
            )

rule()

with st.expander("method notes"):
    folds = load_csv(f"bellwether_{target}_h24{sfx}_folds.csv")
    if len(folds):
        st.dataframe(folds, use_container_width=True, hide_index=True)
    st.markdown(
        """
Rolling-origin backtest, expanding training window, 4 folds of 30 days.

Splits are on the **forecast origin**, not the target timestamp — splitting on the target
lets a training row's outcome land inside the test window. A purge gap of one horizon sits
between train and test for the same reason. The conformal calibration block is taken from
the end of each training window with its own purge gap.

All baselines emit a full predictive distribution, so CRPS comparisons are like-for-like.
The `LightGBM, no weather` row is the same model with weather columns dropped, which is
what makes the gap attributable to the meteorology rather than to a change of estimator.
"""
    )
