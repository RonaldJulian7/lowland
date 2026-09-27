"""Artifact loading and chart plumbing shared by the app pages."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from lowland.config import QUANTILES, REPORTS_DIR
from lowland.viz import Theme

QCOLS = [f"q{q:.2f}" for q in QUANTILES]

# Internal keys are run-together so they survive as filenames and dict keys. Nobody wants
# to read "LightGBM+Mondrian" off a chart axis, so they get spaced out on the way to the UI.
LABELS = {
    "Climatology": "climatology",
    "SeasonalNaive": "seasonal naive",
    "QuantileAR": "linear QR",
    "LightGBM": "LightGBM",
    "LightGBM-NoWeather": "LightGBM, no weather",
    "LightGBM+CQR": "LightGBM + conformal",
    "LightGBM+Mondrian": "LightGBM + Mondrian",
    "DeepTFT": "TFT",
    "DeepTFT+CQR": "TFT + conformal",
    "Ensemble": "ensemble",
    "PerfectForesight": "perfect foresight",
    "Threshold": "threshold rule",
    "DeterministicMPC": "MPC, point forecast",
    "StochasticMPC": "MPC, scenarios",
    "PPO": "RL agent",
}


def label(key: str) -> str:
    return LABELS.get(key, key)


def relabel(obj):
    """Swap internal keys for display labels on a Series or DataFrame index."""
    out = obj.copy()
    out.index = [label(str(i)) for i in out.index]
    return out


def chart(fig, th: Theme, *, table: pd.DataFrame | None = None, key: str | None = None) -> None:
    """Render a Plotly figure with an expandable table beneath it.

    The table is not decoration. Ochre sits below 3:1 against the light paper, and the
    rule covering that case requires visible labels *or* a table view. Shipping the table
    also makes every chart readable by a screen reader and lets someone check a value
    rather than estimate it off an axis.
    """
    st.plotly_chart(
        fig, use_container_width=True, key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    if table is not None and len(table):
        with st.expander("view as table"):
            st.dataframe(table, use_container_width=True, height=min(400, 44 + 28 * len(table)))


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = REPORTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_panel_cached() -> pd.DataFrame:
    from lowland.dataset import load_panel

    return load_panel()


def missing_artifact(what: str, command: str) -> None:
    """Explain how to generate a missing artifact instead of failing silently."""
    st.warning(f"**{what}** has not been generated yet. Run:\n\n```bash\n{command}\n```")


def predictions_frame(target: str, model: str | None = None, suffix: str = "") -> pd.DataFrame:
    """Load Bellwether's pooled out-of-sample predictions for one target."""
    df = load_parquet(f"bellwether_{target}_h24{suffix}_predictions.parquet")
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if model:
        df = df[df["model"] == model]
    return df.sort_values("timestamp")
