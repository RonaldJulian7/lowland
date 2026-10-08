"""Feature construction for direct multi-horizon forecasting.

One model per horizon, predicting y[t] from what was known at t - h. Recursive
forecasting would be cheaper but compounds its own errors, and the error distribution
at step 20 then depends on the whole path taken to get there, which makes calibrated
intervals a nightmare.

The rule that matters: autoregressive features may only touch y at t - h or earlier;
calendar and weather may use their value at t, since both are known ahead. Every lag is
clamped to >= h in _autoregressive_block so it can't be got wrong elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import holidays as pyholidays
import numpy as np
import pandas as pd

from lowland.config import TZ_LOCAL
from lowland.utils import get_logger

log = get_logger(__name__)

#: Weather columns treated as known at forecast time. In operation these come from NWP,
#: which is available well beyond the 48 h horizon used here.
KNOWN_FUTURE_WEATHER = (
    "temp_pop",
    "hdh",
    "cdh",
    "temp_pop_ema24",
    "hdh_ema24",
    "humidity_pop",
    "wind100_off",
    "wind100_on",
    "wind_power_proxy_off",
    "wind_power_proxy_on",
    "ghi_pop",
    "dni_pop",
    "cloud_pop",
    "solar_proxy",
)

#: Seasonal lags, in hours. One day, two days, one week and two weeks capture the
#: dominant repeating structure in both load and price.
SEASONAL_LAGS = (24, 48, 168, 336)

#: Rolling-window lengths in hours for level and dispersion features.
ROLLING_WINDOWS = (24, 168, 720)


@dataclass
class FeatureSpec:
    """Configuration for :func:`make_supervised`."""

    target: str = "residual_load"
    horizon: int = 24
    use_weather: bool = True
    use_calendar: bool = True
    use_autoregressive: bool = True
    #: Additional exogenous panel columns to expose as known-future regressors.
    extra_known: tuple[str, ...] = ()
    #: Additional panel columns to expose as lagged (unknown-future) regressors.
    extra_lagged: tuple[str, ...] = ()
    seasonal_lags: tuple[int, ...] = SEASONAL_LAGS
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS
    fourier_orders: dict[str, int] = field(
        default_factory=lambda: {"daily": 4, "weekly": 3, "yearly": 4}
    )


# --------------------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------------------


def _nl_holiday_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Dutch public-holiday indicators, plus proximity effects.

    The day before and after a public holiday behave differently from both a normal
    weekday and the holiday itself -- industrial load often drops on the bridging day --
    so those are flagged separately rather than folded into one binary.
    """
    local = index.tz_convert(TZ_LOCAL)
    years = sorted(set(local.year))
    nl = pyholidays.country_holidays("NL", years=years)

    dates = pd.Series(local.date, index=index)
    is_hol = dates.map(lambda d: d in nl).astype(int)

    day = pd.Series(pd.to_datetime(local.date), index=index)
    hol_dates = {pd.Timestamp(d) for d in nl}
    is_bridge = day.map(
        lambda d: int(
            (d + pd.Timedelta(days=1)) in hol_dates or (d - pd.Timedelta(days=1)) in hol_dates
        )
    )

    # The last two weeks of December behave like an extended holiday in Dutch industrial
    # demand and are poorly captured by the public-holiday list alone.
    is_xmas = ((local.month == 12) & (local.day >= 20)).astype(int)

    return pd.DataFrame(
        {"is_holiday": is_hol, "is_bridge_day": is_bridge, "is_xmas_period": is_xmas},
        index=index,
    )


def _fourier(index: pd.DatetimeIndex, period_h: float, order: int, label: str) -> pd.DataFrame:
    """Fourier basis of the given order for a seasonality of ``period_h`` hours.

    Fourier terms let a tree ensemble express smooth periodicity that raw integer
    hour-of-day cannot, and let the neural model share parameters across neighbouring
    hours instead of learning 24 unrelated offsets.
    """
    # Hours since epoch measured on the *local wall clock*, so the daily phase stays
    # aligned across the DST switch instead of jumping by an hour twice a year.
    naive_local = index.tz_convert(TZ_LOCAL).tz_localize(None)
    t = naive_local.astype("int64").to_numpy() / 1e9 / 3600.0
    out = {}
    for k in range(1, order + 1):
        ang = 2 * np.pi * k * t / period_h
        out[f"fourier_{label}_sin{k}"] = np.sin(ang)
        out[f"fourier_{label}_cos{k}"] = np.cos(ang)
    return pd.DataFrame(out, index=index)


def calendar_features(index: pd.DatetimeIndex, spec: FeatureSpec) -> pd.DataFrame:
    """Full calendar block: local clock fields, holidays and Fourier seasonality."""
    local = index.tz_convert(TZ_LOCAL)
    base = pd.DataFrame(
        {
            "hour": local.hour,
            "dayofweek": local.dayofweek,
            "month": local.month,
            "dayofyear": local.dayofyear,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            # A linear trend lets the models absorb slow structural change (efficiency
            # gains, electrification) instead of attributing it to weather.
            "time_trend": (index - index.min()).total_seconds() / (365.25 * 24 * 3600),
        },
        index=index,
    )
    blocks = [base, _nl_holiday_flags(index)]
    orders = spec.fourier_orders
    if orders.get("daily"):
        blocks.append(_fourier(index, 24.0, orders["daily"], "d"))
    if orders.get("weekly"):
        blocks.append(_fourier(index, 168.0, orders["weekly"], "w"))
    if orders.get("yearly"):
        blocks.append(_fourier(index, 8766.0, orders["yearly"], "y"))
    return pd.concat(blocks, axis=1)


# --------------------------------------------------------------------------------------
# Autoregressive block
# --------------------------------------------------------------------------------------


def _autoregressive_block(
    series: pd.Series, horizon: int, spec: FeatureSpec, prefix: str
) -> pd.DataFrame:
    """Lag/rolling features for ``series``, clamped so nothing leaks past the origin.

    Every lag is at least ``horizon`` hours. Seasonal lags are rounded *up* to the next
    multiple of 24 at or beyond the horizon so that, for example, a 48 h forecast uses
    the same hour two days back rather than an unavailable one-day-back value.
    """
    out: dict[str, pd.Series] = {}

    # The most recent observation available at the origin.
    out[f"{prefix}_lag{horizon}"] = series.shift(horizon)

    for lag in spec.seasonal_lags:
        eff = lag if lag >= horizon else int(np.ceil(horizon / 24.0) * 24)
        out[f"{prefix}_slag{eff}"] = series.shift(eff)

    # Rolling statistics computed on data up to the origin only.
    at_origin = series.shift(horizon)
    for w in spec.rolling_windows:
        out[f"{prefix}_rmean{w}"] = at_origin.rolling(w, min_periods=max(2, w // 4)).mean()
        out[f"{prefix}_rstd{w}"] = at_origin.rolling(w, min_periods=max(2, w // 4)).std()
    out[f"{prefix}_rmin168"] = at_origin.rolling(168, min_periods=24).min()
    out[f"{prefix}_rmax168"] = at_origin.rolling(168, min_periods=24).max()

    # Same-hour-yesterday differenced against same-hour-last-week isolates whether the
    # recent level shift is a one-off or a persistent regime change.
    d1 = out.get(f"{prefix}_slag24", out[f"{prefix}_lag{horizon}"])
    d7 = out.get(f"{prefix}_slag168")
    if d7 is not None:
        out[f"{prefix}_wow_delta"] = d1 - d7

    frame = pd.DataFrame(out)
    # Deduplicate columns that collapse onto the same effective lag at long horizons.
    return frame.loc[:, ~frame.columns.duplicated()]


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def make_supervised(
    panel: pd.DataFrame, spec: FeatureSpec
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Build ``(X, y, origin_index)`` for one target and one horizon.

    Returns
    -------
    X:
        Feature matrix indexed by *target* timestamp.
    y:
        Target series, aligned to ``X``. Rows where the target is missing are retained
        with ``NaN`` so the same matrix can serve for inference; callers filter with
        :func:`train_mask`.
    origin_index:
        The forecast-origin timestamp for each row, i.e. ``target_time - horizon``. Used
        by the backtester to split on origin rather than on target time, which is the
        only split that reflects what a forecaster actually knew.
    """
    if spec.target not in panel.columns:
        raise KeyError(f"target {spec.target!r} not in panel columns")

    idx = panel.index
    y = panel[spec.target].astype("float64")
    blocks: list[pd.DataFrame] = []

    if spec.use_calendar:
        blocks.append(calendar_features(idx, spec))

    if spec.use_weather:
        wx = [c for c in KNOWN_FUTURE_WEATHER if c in panel.columns]
        w = panel[wx].copy()
        # Weather at the target hour is known from NWP. Short leads and lags around it
        # give the model the local shape of the weather event, not just a point value.
        for c in ("temp_pop", "wind_power_proxy_off", "solar_proxy"):
            if c in w.columns:
                w[f"{c}_lead3"] = panel[c].shift(-3)
                w[f"{c}_lag3"] = panel[c].shift(3)
                w[f"{c}_d24"] = panel[c] - panel[c].shift(24)
        blocks.append(w)

    for c in spec.extra_known:
        if c in panel.columns:
            blocks.append(panel[[c]].rename(columns={c: f"known_{c}"}))

    if spec.use_autoregressive:
        blocks.append(_autoregressive_block(y, spec.horizon, spec, prefix="y"))
        for c in spec.extra_lagged:
            if c in panel.columns:
                blocks.append(
                    _autoregressive_block(
                        panel[c].astype("float64"), spec.horizon, spec, prefix=c
                    )
                )

    X = pd.concat(blocks, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    # Interactions the tree models would otherwise need many splits to approximate.
    if {"hdh", "is_weekend"} <= set(X.columns):
        X["hdh_x_weekend"] = X["hdh"] * X["is_weekend"]
    if {"hdh", "hour"} <= set(X.columns):
        X["hdh_x_hour"] = X["hdh"] * X["hour"]
    if {"wind_power_proxy_off", "wind_power_proxy_on"} <= set(X.columns):
        # Fleet-wide wind capacity factor, weighted towards offshore which now dominates
        # Dutch installed capacity.
        X["wind_fleet_cf"] = (
            0.65 * X["wind_power_proxy_off"] + 0.35 * X["wind_power_proxy_on"]
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    origin_index = idx - pd.Timedelta(hours=spec.horizon)
    return X, y, origin_index


def train_mask(X: pd.DataFrame, y: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Rows usable for fitting: target present, quality flag set, features not all-NaN."""
    ok = y.notna()
    if "is_complete" in panel.columns:
        ok &= panel["is_complete"] == 1
    # Early rows lack their longest rolling window; drop rather than impute, since an
    # imputed 30-day mean at the very start of the sample is pure noise.
    ok &= X.notna().mean(axis=1) > 0.80
    return ok


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Group feature names for grouped permutation importance and app display."""
    groups: dict[str, list[str]] = {
        "calendar": [],
        "seasonality": [],
        "weather_temp": [],
        "weather_wind": [],
        "weather_solar": [],
        "autoregressive": [],
        "other": [],
    }
    for c in columns:
        if c.startswith("fourier_"):
            groups["seasonality"].append(c)
        elif c in {"hour", "dayofweek", "month", "dayofyear", "is_weekend", "time_trend"} or c.startswith(
            ("is_holiday", "is_bridge", "is_xmas")
        ):
            groups["calendar"].append(c)
        elif "temp" in c or c.startswith(("hdh", "cdh", "humidity")):
            groups["weather_temp"].append(c)
        elif "wind" in c:
            groups["weather_wind"].append(c)
        elif any(k in c for k in ("solar", "ghi", "dni", "cloud")):
            groups["weather_solar"].append(c)
        elif c.startswith("y_") or "_lag" in c or "_slag" in c or "_rmean" in c or "_rstd" in c:
            groups["autoregressive"].append(c)
        else:
            groups["other"].append(c)
    return {k: v for k, v in groups.items() if v}
