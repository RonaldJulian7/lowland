"""Parquet cache with a provenance manifest.

Everything written through write_cached() gets an entry in data/cache/_manifest.json --
source, fetch time, row count, covered range. Means the pipeline reruns offline from
committed snapshots, and any figure can be traced back to a specific retrieval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lowland.config import CACHE_DIR
from lowland.utils import get_logger

log = get_logger(__name__)

MANIFEST_PATH = CACHE_DIR / "_manifest.json"


@dataclass
class CacheEntry:
    """Provenance record for one cached dataset."""

    key: str
    path: str
    source: str
    fetched_at: str
    n_rows: int
    n_cols: int
    index_start: str | None
    index_end: str | None
    params_hash: str
    params: dict[str, Any]


def _load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("cache manifest was corrupt; starting a fresh one")
        return {}


def _save_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def params_hash(params: dict[str, Any]) -> str:
    """Stable short hash of a request-parameter dictionary."""
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def read_cached(key: str) -> pd.DataFrame | None:
    """Return the cached frame for ``key``, or ``None`` when it has not been fetched."""
    path = cache_path(key)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cached(
    key: str,
    df: pd.DataFrame,
    *,
    source: str,
    params: dict[str, Any] | None = None,
) -> Path:
    """Persist ``df`` under ``key`` and record its provenance in the manifest."""
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)

    idx_start = idx_end = None
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        idx_start, idx_end = str(df.index.min()), str(df.index.max())

    params = params or {}
    entry = CacheEntry(
        key=key,
        path=str(path.relative_to(CACHE_DIR.parents[1])),
        source=source,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        index_start=idx_start,
        index_end=idx_end,
        params_hash=params_hash(params),
        params=params,
    )
    manifest = _load_manifest()
    manifest[key] = asdict(entry)
    _save_manifest(manifest)
    log.info("cached %s  rows=%s  cols=%s  -> %s", key, len(df), df.shape[1], path.name)
    return path


CHUNK_DIR = CACHE_DIR / "chunks"


def chunk_cached(
    prefix: str,
    chunks: list[tuple[Any, Any]],
    fetch_one: "Callable[[Any, Any], pd.DataFrame]",
    *,
    source: str,
) -> pd.DataFrame:
    """Fetch a long history chunk by chunk, persisting each piece as it arrives.

    Long backfills against rate-limited public APIs fail partway through as a matter of
    course. Without per-chunk persistence a failure at chunk 23 of 24 throws away every
    earlier request and the retry hits the same quota wall again, which is exactly how a
    backfill becomes unfinishable. Caching each chunk makes the operation *resumable*: a
    rerun re-requests only what is genuinely missing.
    """
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    fetched = 0

    for c_start, c_end in chunks:
        path = CHUNK_DIR / f"{prefix}_{c_start}_{c_end}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        df = fetch_one(c_start, c_end)
        if df is not None and len(df):
            df.to_parquet(path)
            frames.append(df)
            fetched += 1

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    log.info(
        "%s: %s chunks (%s newly fetched, %s from cache) -> %s rows",
        prefix, len(chunks), fetched, len(chunks) - fetched, len(out),
    )
    return out


def manifest_table() -> pd.DataFrame:
    """Return the manifest as a frame, for display in notebooks and the apps."""
    manifest = _load_manifest()
    if not manifest:
        return pd.DataFrame(
            columns=["key", "source", "fetched_at", "n_rows", "index_start", "index_end"]
        )
    rows = [
        {
            "key": v["key"],
            "source": v["source"],
            "fetched_at": v["fetched_at"],
            "n_rows": v["n_rows"],
            "index_start": v["index_start"],
            "index_end": v["index_end"],
        }
        for v in manifest.values()
    ]
    return pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
