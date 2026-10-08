"""Client for the energy-charts.info open API (Fraunhofer ISE).

The service republishes ENTSO-E Transparency Platform data under CC-BY-4.0 without
requiring an API token, which is what makes this repository reproducible by anyone.

Two endpoints are used:

``/public_power``
    Generation by production type plus ``Load``, ``Residual load`` and renewable-share
    series for a bidding zone, at the native 15-minute resolution.

``/price``
    Day-ahead auction clearing prices for a bidding zone, in EUR/MWh.

Both return a "struct of arrays" payload: a single ``unix_seconds`` vector plus parallel
value vectors, which this module transposes into tidy, UTC-indexed frames.
"""

from __future__ import annotations

import pandas as pd

from lowland.config import HISTORY_START
from lowland.io.cache import chunk_cached, read_cached, write_cached
from lowland.io.http import get_json
from lowland.utils import daterange_chunks, get_logger, to_utc_index

log = get_logger(__name__)

BASE = "https://api.energy-charts.info"
LICENSE = "CC BY 4.0 - energy-charts.info (Fraunhofer ISE), derived from ENTSO-E"

#: Series in ``/public_power`` that are aggregates rather than generation units. They are
#: kept but flagged so that "sum over production types" never double counts.
DERIVED_SERIES = {
    "Load",
    "Residual load",
    "Renewable share of load",
    "Renewable share of generation",
    "Cross border electricity trading",
}


def _slug(name: str) -> str:
    """Turn an upstream series name into a stable snake_case column name."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
        .strip("_")
    )


#: Seconds between consecutive calls. The service rate-limits over a rolling window, and
#: a full 2015-onwards backfill is ~50 requests; pacing them is the difference between a
#: run that completes and one that trips a 429 two thirds of the way through.
MIN_INTERVAL_S = 4.0


def fetch_public_power(
    start: str, end: str, country: str = "nl", *, chunk_days: int = 180
) -> pd.DataFrame:
    """Fetch generation-by-type and load series for ``country`` over ``[start, end]``.

    Returns a UTC-indexed frame at 15-minute resolution with one column per production
    type (MW), plus ``load``, ``residual_load`` and the two renewable-share columns (%).
    Requests are chunked and each chunk is cached, so the call is resumable.
    """

    def _one(c_start, c_end) -> pd.DataFrame:
        payload = get_json(
            f"{BASE}/public_power",
            {"country": country, "start": str(c_start), "end": str(c_end)},
            min_interval_s=MIN_INTERVAL_S,
        )
        idx = to_utc_index(payload["unix_seconds"])
        data = {
            _slug(pt["name"]): pd.Series(pt["data"], index=idx, dtype="float64")
            for pt in payload["production_types"]
        }
        log.info("public_power %s %s..%s  rows=%s", country, c_start, c_end, len(idx))
        return pd.DataFrame(data)

    chunks = list(daterange_chunks(start, end, days=chunk_days))
    return chunk_cached(f"ec_power_{country}", chunks, _one, source=LICENSE)


def fetch_day_ahead_price(
    start: str, end: str, bzn: str = "NL", *, chunk_days: int = 180
) -> pd.DataFrame:
    """Fetch day-ahead clearing prices (EUR/MWh) for bidding zone ``bzn``."""

    def _one(c_start, c_end) -> pd.DataFrame:
        payload = get_json(
            f"{BASE}/price",
            {"bzn": bzn, "start": str(c_start), "end": str(c_end)},
            min_interval_s=MIN_INTERVAL_S,
        )
        idx = to_utc_index(payload["unix_seconds"])
        log.info("price %s %s..%s  rows=%s", bzn, c_start, c_end, len(idx))
        return pd.DataFrame({"price_da": pd.Series(payload["price"], index=idx)})

    chunks = list(daterange_chunks(start, end, days=chunk_days))
    return chunk_cached(f"ec_price_{bzn}", chunks, _one, source=LICENSE)


def load_power(
    start: str = HISTORY_START, end: str | None = None, *, refresh: bool = False
) -> pd.DataFrame:
    """Cached accessor for the generation/load feed."""
    end = end or str(pd.Timestamp.utcnow().date())
    key = f"energy_charts_power_nl_{start}_{end}"
    if not refresh:
        cached = read_cached(key)
        if cached is not None:
            return cached
    df = fetch_public_power(start, end)
    write_cached(key, df, source=LICENSE, params={"start": start, "end": end, "country": "nl"})
    return df


def load_price(
    start: str = HISTORY_START, end: str | None = None, *, refresh: bool = False
) -> pd.DataFrame:
    """Cached accessor for the day-ahead price feed."""
    end = end or str(pd.Timestamp.utcnow().date())
    key = f"energy_charts_price_nl_{start}_{end}"
    if not refresh:
        cached = read_cached(key)
        if cached is not None:
            return cached
    df = fetch_day_ahead_price(start, end)
    write_cached(key, df, source=LICENSE, params={"start": start, "end": end, "bzn": "NL"})
    return df
