"""Open-Meteo client: ERA5 reanalysis plus operational forecasts.

Both endpoints are needed. The archive is ERA5 reanalysis -- great quality, but published
about five days late, so it can't cover yesterday. The forecast endpoint back-fills that
gap and runs 16 days ahead, which is what makes a genuine forward forecast possible
rather than a backtest-only exercise.

Training on reanalysis and predicting on NWP is a small train/serve shift. That's the
situation any operational forecaster is in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lowland.config import ALL_SITES, HISTORY_START, HOURLY_WEATHER_VARS, WeatherSite
from lowland.io.cache import chunk_cached, read_cached, write_cached
from lowland.io.http import get_json
from lowland.utils import daterange_chunks, get_logger

log = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LICENSE = "CC BY 4.0 - Open-Meteo.com (ERA5 reanalysis / operational NWP)"


def _payload_to_frame(payload: dict, site: WeatherSite) -> pd.DataFrame:
    """Convert one Open-Meteo location payload into a UTC-indexed frame."""
    hourly = payload["hourly"]
    # Explicit DatetimeIndex, not a Series: see the note in lowland.utils.to_utc_index.
    idx = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="timestamp")
    cols = {
        f"{site.name}__{var}": pd.Series(hourly[var], index=idx, dtype="float64")
        for var in HOURLY_WEATHER_VARS
        if var in hourly
    }
    df = pd.DataFrame(cols)
    df.index.name = "timestamp"
    return df


def _fetch_sites(
    url: str,
    sites: tuple[WeatherSite, ...],
    extra_params: dict,
) -> pd.DataFrame:
    """Request all ``sites`` in one call and return the horizontally joined frame.

    Open-Meteo accepts comma-separated coordinate lists and then returns a JSON *array*
    of per-location payloads, in the same order as the request.
    """
    params = {
        "latitude": [s.lat for s in sites],
        "longitude": [s.lon for s in sites],
        "hourly": list(HOURLY_WEATHER_VARS),
        "timezone": "UTC",
        **extra_params,
    }
    payload = get_json(url, params, min_interval_s=1.2)
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(sites):
        raise ValueError(
            f"Open-Meteo returned {len(payloads)} locations for {len(sites)} requested sites"
        )
    frames = [_payload_to_frame(p, s) for p, s in zip(payloads, sites)]
    return pd.concat(frames, axis=1)


def fetch_archive(
    start: str,
    end: str,
    sites: tuple[WeatherSite, ...] = ALL_SITES,
    *,
    chunk_days: int = 1500,
) -> pd.DataFrame:
    """Fetch ERA5 reanalysis for ``sites`` over ``[start, end]``."""

    def _one(c_start, c_end) -> pd.DataFrame:
        df = _fetch_sites(
            ARCHIVE_URL, sites, {"start_date": str(c_start), "end_date": str(c_end)}
        )
        log.info("open-meteo archive %s..%s  rows=%s cols=%s", c_start, c_end, *df.shape)
        return df

    chunks = list(daterange_chunks(start, end, days=chunk_days))
    return chunk_cached("om_archive", chunks, _one, source=LICENSE)


def fetch_forecast(
    sites: tuple[WeatherSite, ...] = ALL_SITES,
    *,
    past_days: int = 92,
    forecast_days: int = 16,
) -> pd.DataFrame:
    """Fetch recent-past plus forward NWP for ``sites``.

    ``past_days`` covers the reanalysis publication lag; ``forecast_days`` supplies the
    known-future covariates used at inference time.
    """
    df = _fetch_sites(
        FORECAST_URL,
        sites,
        {"past_days": past_days, "forecast_days": forecast_days},
    )
    log.info("open-meteo forecast rows=%s cols=%s", *df.shape)
    return df.sort_index()


def national_aggregates(weather: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-site weather into physically meaningful national indices.

    Three families of index are produced.

    ``temp_pop`` and the degree-day terms
        Population-weighted temperature over the demand centres, plus heating- and
        cooling-degree hours. Dutch demand responds to temperature asymmetrically and
        non-linearly, so exposing the degree terms directly saves the models from having
        to rediscover the kink at roughly 15 degrees Celsius.

    ``wind_power_proxy_{off,on}``
        Capacity-weighted mean of the *cubed* 100 m wind speed, clipped by a logistic
        power curve. Turbine output scales with the cube of wind speed below rated speed
        and is constant above it, so a linear wind-speed feature is badly mis-specified.

    ``solar_proxy``
        Population-weighted shortwave irradiance, which tracks PV output far more closely
        than cloud cover does.
    """
    out = pd.DataFrame(index=weather.index)

    def wmean(sites, var: str) -> pd.Series:
        cols = [f"{s.name}__{var}" for s in sites if f"{s.name}__{var}" in weather.columns]
        if not cols:
            return pd.Series(np.nan, index=weather.index)
        w = np.array([s.weight for s in sites if f"{s.name}__{var}" in weather.columns])
        w = w / w.sum()
        return (weather[cols].to_numpy() * w).sum(axis=1)

    load_sites = tuple(s for s in ALL_SITES if s.kind == "load")
    off_sites = tuple(s for s in ALL_SITES if s.kind == "wind_off")
    on_sites = tuple(s for s in ALL_SITES if s.kind == "wind_on")

    # --- demand-side temperature indices -------------------------------------------
    out["temp_pop"] = wmean(load_sites, "temperature_2m")
    out["humidity_pop"] = wmean(load_sites, "relative_humidity_2m")
    # 15.5 C is the conventional European base temperature for heating degree days.
    out["hdh"] = (15.5 - out["temp_pop"]).clip(lower=0)
    out["cdh"] = (out["temp_pop"] - 22.0).clip(lower=0)
    # Buildings have thermal mass: yesterday's cold still drives today's heating load.
    out["temp_pop_ema24"] = out["temp_pop"].ewm(halflife=24, min_periods=1).mean()
    out["hdh_ema24"] = out["hdh"].ewm(halflife=24, min_periods=1).mean()

    # --- wind power proxies ----------------------------------------------------------
    for label, sites in (("off", off_sites), ("on", on_sites)):
        ws = wmean(sites, "wind_speed_100m")  # km/h from Open-Meteo
        ws_ms = pd.Series(ws, index=weather.index) / 3.6
        out[f"wind100_{label}"] = ws_ms
        out[f"wind_power_proxy_{label}"] = _power_curve(ws_ms)

    # --- solar proxy -----------------------------------------------------------------
    out["ghi_pop"] = wmean(load_sites, "shortwave_radiation")
    out["dni_pop"] = wmean(load_sites, "direct_normal_irradiance")
    out["cloud_pop"] = wmean(load_sites, "cloud_cover")
    out["solar_proxy"] = out["ghi_pop"].clip(lower=0)

    return out


def _power_curve(
    wind_ms: pd.Series,
    cut_in: float = 3.0,
    rated: float = 12.5,
    cut_out: float = 25.0,
) -> pd.Series:
    """Normalised turbine power curve, returning capacity factor in ``[0, 1]``.

    Below ``cut_in`` output is zero; between cut-in and ``rated`` it follows the cubic
    law normalised to reach 1 at rated speed; above rated it is flat until ``cut_out``,
    beyond which the turbine feathers and output drops to zero. This is a deliberately
    simple aggregate curve: across a whole national fleet the sharp individual cut-out is
    smeared out, which the models absorb through the raw ``wind100_*`` feature kept
    alongside it.
    """
    w = wind_ms.clip(lower=0)
    cf = ((w**3 - cut_in**3) / (rated**3 - cut_in**3)).clip(lower=0, upper=1)
    cf = cf.where(w >= cut_in, 0.0)
    cf = cf.where(w <= cut_out, 0.0)
    return cf


def load_weather(
    start: str = HISTORY_START, end: str | None = None, *, refresh: bool = False
) -> pd.DataFrame:
    """Cached accessor for the reanalysis history."""
    end = end or str((pd.Timestamp.utcnow() - pd.Timedelta(days=6)).date())
    key = f"open_meteo_archive_{start}_{end}"
    if not refresh:
        cached = read_cached(key)
        if cached is not None:
            return cached
    df = fetch_archive(start, end)
    write_cached(key, df, source=LICENSE, params={"start": start, "end": end})
    return df


def load_weather_forecast(*, refresh: bool = False, past_days: int = 92) -> pd.DataFrame:
    """Cached accessor for the operational NWP window.

    Cached under a date-stamped key so that a rerun on the same day is free, while a
    rerun tomorrow correctly pulls a fresh forecast.
    """
    stamp = str(pd.Timestamp.utcnow().date())
    key = f"open_meteo_forecast_{stamp}"
    if not refresh:
        cached = read_cached(key)
        if cached is not None:
            return cached
    df = fetch_forecast(past_days=past_days)
    write_cached(key, df, source=LICENSE, params={"as_of": stamp, "past_days": past_days})
    return df
