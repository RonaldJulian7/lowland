"""Construction of the master hourly panel for the Dutch power system.

The panel joins three feeds onto a single UTC hourly index:

* generation by production type, load and residual load (energy-charts / ENTSO-E);
* day-ahead clearing price (energy-charts / ENTSO-E);
* weather, as per-site reanalysis collapsed into national physical indices (Open-Meteo).

It is the single input to all three projects, which keeps their results directly
comparable and means a data fix propagates everywhere at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lowland.config import (
    CONGESTION_PERCENTILE,
    HISTORY_START,
    PROCESSED_DIR,
)
from lowland.io import energy_charts, open_meteo
from lowland.io.cache import write_cached
from lowland.utils import get_logger, to_hourly

log = get_logger(__name__)

PANEL_PATH = PROCESSED_DIR / "panel_hourly.parquet"

#: Columns carried through from the generation feed. Kept explicit so that an upstream
#: schema change surfaces as a loud KeyError rather than as silently missing features.
GENERATION_COLS = [
    "nuclear",
    "biomass",
    "fossil_hard_coal",
    "fossil_gas",
    "fossil_oil",
    "waste",
    "others",
    "wind_offshore",
    "wind_onshore",
    "solar",
    "hydro_run_of_river",
]

DERIVED_COLS = [
    "load",
    "residual_load",
    "cross_border_electricity_trading",
    "renewable_share_of_load",
    "renewable_share_of_generation",
]


def build_panel(
    start: str = HISTORY_START,
    end: str | None = None,
    *,
    refresh: bool = False,
    include_forecast_window: bool = True,
) -> pd.DataFrame:
    """Build (and cache) the master hourly panel.

    Parameters
    ----------
    start, end:
        Inclusive date bounds. ``end`` defaults to today (UTC).
    refresh:
        Bypass the on-disk caches of the individual feeds and re-download.
    include_forecast_window:
        Append the operational NWP window so the panel extends into the future with
        weather present and targets missing. Downstream code treats rows with a missing
        target as the inference set, which is exactly what a live forecast needs.
    """
    end = end or str(pd.Timestamp.utcnow().date())

    # ---- power and price -------------------------------------------------------------
    power = energy_charts.load_power(start, end, refresh=refresh)
    price = energy_charts.load_price(start, end, refresh=refresh)

    power_h = to_hourly(power, how="mean")
    price_h = to_hourly(price, how="mean")

    keep = [c for c in GENERATION_COLS + DERIVED_COLS if c in power_h.columns]
    missing = sorted(set(DERIVED_COLS) - set(power_h.columns))
    if missing:
        log.warning("derived columns absent from upstream feed: %s", missing)
    power_h = power_h[keep]

    # ---- weather ---------------------------------------------------------------------
    wx_hist = open_meteo.load_weather(start, refresh=refresh)
    wx_parts = [wx_hist]
    if include_forecast_window:
        wx_fc = open_meteo.load_weather_forecast(refresh=refresh)
        # Reanalysis wins wherever both exist: it is the better estimate of what actually
        # happened, and using it for training avoids learning NWP bias as if it were signal.
        wx_fc = wx_fc.loc[~wx_fc.index.isin(wx_hist.index)]
        wx_parts.append(wx_fc)
    weather = pd.concat(wx_parts).sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]

    agg = open_meteo.national_aggregates(weather)

    # ---- join --------------------------------------------------------------------------
    panel = agg.join(power_h, how="outer").join(price_h, how="outer").sort_index()
    # Defensive: an upstream schema quirk or a parquet round-trip must never be able to
    # hand downstream code a panel without working datetime accessors.
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index, utc=True))
    panel.index.name = "timestamp"

    panel = _add_derived_targets(panel)
    panel = _flag_quality(panel)

    log.info(
        "panel built: %s rows x %s cols, %s .. %s",
        len(panel),
        panel.shape[1],
        panel.index.min(),
        panel.index.max(),
    )
    return panel


def _add_derived_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Add modelling targets that are not supplied directly upstream."""
    df = panel.copy()

    vre_cols = [c for c in ("wind_offshore", "wind_onshore", "solar") if c in df.columns]
    if vre_cols:
        df["vre_total"] = df[vre_cols].sum(axis=1, min_count=1)

    # Recompute residual load rather than trusting the upstream aggregate blindly; where
    # both exist they should agree, and the check is cheap.
    if "load" in df.columns and "vre_total" in df.columns:
        df["residual_load_calc"] = df["load"] - df["vre_total"]
        if "residual_load" in df.columns:
            gap = (df["residual_load"] - df["residual_load_calc"]).abs()
            if gap.notna().any():
                log.info(
                    "residual-load reconciliation: median |diff| = %.1f MW",
                    float(gap.median(skipna=True)),
                )
        else:
            df["residual_load"] = df["residual_load_calc"]

    # VRE penetration is the key regressor for the merit-order analysis in project 3.
    if {"vre_total", "load"} <= set(df.columns):
        df["vre_share"] = (df["vre_total"] / df["load"].replace(0, np.nan)).clip(0, 2)

    # Thermal fleet output drives both emissions and marginal cost.
    thermal = [
        c for c in ("fossil_gas", "fossil_hard_coal", "fossil_oil", "waste") if c in df.columns
    ]
    if thermal:
        df["thermal_total"] = df[thermal].sum(axis=1, min_count=1)

    # Hour-on-hour ramp of residual load is what actually stresses dispatchable plant.
    if "residual_load" in df.columns:
        df["residual_ramp"] = df["residual_load"].diff()

    return df


def detect_provisional_tail(
    panel: pd.DataFrame,
    *,
    ratio_threshold: float = 0.88,
    lookback_days: int = 150,
) -> pd.Timestamp | None:
    """Find the date from which the upstream feed is still provisional.

    ENTSO-E actuals are published quickly but settled slowly: for the most recent weeks
    not every unit has reported, so aggregate load and generation are systematically too
    low. On the snapshot this repository was built against, Dutch load drops from roughly
    12,500 MW to 8,000-10,500 MW overnight while the day-of-year reference is unchanged --
    a ~25% shortfall that is an artefact of settlement, not a demand collapse.

    Training through that tail is survivable; *scoring* on it is not. It inflates test
    error several-fold and destroys interval coverage, and it would do so in a way that
    looks like genuine model failure. Worse, the cutoff moves every time the data is
    refreshed, so hard-coding a date would silently rot.

    Detection compares each day's mean load against a day-of-year climatology built from
    history *older* than ``lookback_days`` (so the reference cannot be contaminated by the
    very tail being tested), then returns the start of the contiguous run of
    below-threshold days ending at the last observation. Requiring the run to reach the
    end is what distinguishes a settlement tail from an ordinary low-demand day such as
    Christmas.

    Returns ``None`` when the feed looks fully settled.
    """
    if "load" not in panel.columns:
        return None
    load = panel["load"].dropna()
    if load.empty:
        return None

    daily = load.resample("D").mean().dropna()
    if len(daily) < 400:  # need a couple of years before a climatology means anything
        return None

    end = daily.index.max()
    ref_src = daily[daily.index < end - pd.Timedelta(days=lookback_days)]
    if len(ref_src) < 365:
        return None

    by_doy = ref_src.groupby(ref_src.index.dayofyear).median()
    by_doy = by_doy.reindex(range(1, 367)).interpolate(limit_direction="both")
    # Smooth circularly so 31 December and 1 January are neighbours.
    tripled = pd.concat([by_doy, by_doy, by_doy], ignore_index=True)
    smoothed = tripled.rolling(15, center=True, min_periods=1).mean()
    ref = pd.Series(
        smoothed.iloc[len(by_doy) : 2 * len(by_doy)].to_numpy(), index=by_doy.index
    )

    ratio = daily / daily.index.dayofyear.map(ref)
    recent = ratio[ratio.index > end - pd.Timedelta(days=lookback_days)].dropna()
    if recent.empty or recent.iloc[-1] >= ratio_threshold:
        return None

    below = recent < ratio_threshold
    cutoff = recent.index[-1]
    for ts in below.index[::-1]:
        if not below.loc[ts]:
            break
        cutoff = ts

    log.warning(
        "provisional data detected from %s onward (%s days, mean load ratio %.2f of "
        "day-of-year reference); these rows are excluded from fitting and scoring",
        cutoff.date(),
        int((end - cutoff).days) + 1,
        float(recent.loc[cutoff:].mean()),
    )
    return cutoff


def _flag_quality(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach data-quality flags instead of silently imputing.

    ``is_complete`` marks rows where every modelling target is present. Rows failing it
    are excluded from training and from scoring, but kept in the panel so that gaps stay
    visible in the apps rather than being papered over.
    """
    df = panel.copy()
    core = [c for c in ("load", "residual_load", "price_da") if c in df.columns]
    df["is_complete"] = df[core].notna().all(axis=1).astype(int) if core else 0

    # Rows in the provisional settlement tail are present but wrong. They are kept in the
    # panel and flagged, so the apps can show them clearly labelled, but excluded from
    # anything that fits or scores a model.
    cutoff = detect_provisional_tail(df)
    df["is_provisional"] = 0
    if cutoff is not None:
        mask = df.index >= cutoff
        df.loc[mask, "is_provisional"] = 1
        df.loc[mask, "is_complete"] = 0

    # A day-ahead price that is exactly zero for a long run is far more likely to be a
    # feed artefact than a real market outcome; genuine zero/negative prices are isolated.
    if "price_da" in df.columns:
        zero_run = (df["price_da"] == 0).astype(int)
        df["price_zero_run"] = (
            zero_run.groupby((zero_run != zero_run.shift()).cumsum()).transform("sum") * zero_run
        )
        suspicious = int((df["price_zero_run"] > 6).sum())
        if suspicious:
            log.warning("%s hours sit in a zero-price run longer than 6h", suspicious)

    return df


def congestion_threshold(panel: pd.DataFrame, percentile: float = CONGESTION_PERCENTILE) -> float:
    """Empirical residual-load level defining system stress.

    Defined as a high percentile of the observed distribution rather than a fixed MW
    figure, so that the definition tracks structural change in the fleet instead of going
    stale as renewables grow.
    """
    rl = panel["residual_load"].dropna()
    if rl.empty:
        from lowland.config import CONGESTION_THRESHOLD_MW_FALLBACK

        return CONGESTION_THRESHOLD_MW_FALLBACK
    return float(rl.quantile(percentile))


def save_panel(panel: pd.DataFrame) -> None:
    """Persist the panel to ``data/processed`` and record it in the cache manifest."""
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)
    write_cached(
        "panel_hourly",
        panel,
        source="derived: energy-charts (ENTSO-E) + Open-Meteo",
        params={"rows": len(panel), "cols": int(panel.shape[1])},
    )


def load_panel() -> pd.DataFrame:
    """Load the cached panel, raising a helpful error if it has not been built."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{PANEL_PATH} not found. Run `python -m scripts.build_dataset` first."
        )
    return pd.read_parquet(PANEL_PATH)
