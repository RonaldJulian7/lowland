"""Small shared helpers: logging, seeding, device selection and time handling."""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from rich.logging import RichHandler

from lowland.config import MODEL_FREQ, RANDOM_SEED, TORCH_DEVICE, TZ_LOCAL

_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a Rich-formatted logger, configuring the root handler exactly once."""
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("NLEI_LOGLEVEL", "INFO"),
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and (if importable) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def resolve_device(preference: str = TORCH_DEVICE) -> str:
    """Resolve ``auto`` to ``cuda`` when a GPU is usable, otherwise ``cpu``."""
    import torch

    if preference != "auto":
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def daterange_chunks(
    start: str | date | datetime,
    end: str | date | datetime,
    days: int = 365,
) -> Iterator[tuple[date, date]]:
    """Yield inclusive ``(chunk_start, chunk_end)`` date pairs covering ``[start, end]``.

    Upstream APIs cap the span of a single request, so long histories have to be pulled
    in slices. Chunks are inclusive at both ends; callers are expected to de-duplicate on
    the timestamp index after concatenation.
    """
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    if s > e:
        return
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def to_utc_index(unix_seconds: list[int] | np.ndarray) -> pd.DatetimeIndex:
    """Convert a list of epoch seconds into a tz-aware UTC :class:`~pandas.DatetimeIndex`.

    Constructed explicitly as a ``DatetimeIndex``. Going through ``pd.Series`` and
    relying on pandas to coerce it at assignment time works most of the time and then
    silently yields a plain object ``Index`` in some paths, which strips ``.year``,
    ``.hour`` and every resampling operation downstream.
    """
    return pd.DatetimeIndex(
        pd.to_datetime(np.asarray(unix_seconds), unit="s", utc=True), name="timestamp"
    )


def to_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Down-sample a UTC-indexed frame to the modelling resolution.

    Power flows (MW) and prices are averaged rather than summed: a 15-minute MW value is
    an instantaneous rate, so the hourly mean is the correct energy-preserving aggregate.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("to_hourly expects a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("index must be tz-aware (UTC) before resampling")
    out = getattr(df.resample(MODEL_FREQ), how)()
    return out


def local_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive Dutch-local calendar fields from a UTC index.

    Converting to ``Europe/Amsterdam`` before extracting hour-of-day is essential: demand
    and solar output follow local clock time, not UTC, and the summer/winter offset
    differs by an hour.
    """
    local = index.tz_convert(TZ_LOCAL)
    return pd.DataFrame(
        {
            "hour": local.hour,
            "dayofweek": local.dayofweek,
            "dayofyear": local.dayofyear,
            "month": local.month,
            "year": local.year,
            "is_weekend": (local.dayofweek >= 5).astype(int),
        },
        index=index,
    )


def human_bytes(n: int) -> str:
    """Format a byte count for log output."""
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < step:
            return f"{n:3.1f} {unit}"
        n /= step  # type: ignore[assignment]
    return f"{n:.1f} TiB"
