"""Shared HTTP session with retry, backoff and polite rate limiting."""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from lowland.config import HTTP
from lowland.utils import get_logger

log = get_logger(__name__)


class TransientHTTPError(RuntimeError):
    """Raised for status codes that are worth retrying (429 and 5xx)."""


_LAST_CALL: dict[str, float] = {}


def _throttle(host: str, min_interval_s: float) -> None:
    """Block until at least ``min_interval_s`` has elapsed since the last call to ``host``."""
    last = _LAST_CALL.get(host)
    now = time.monotonic()
    if last is not None:
        wait = min_interval_s - (now - last)
        if wait > 0:
            time.sleep(wait)
    _LAST_CALL[host] = time.monotonic()


@retry(
    retry=retry_if_exception_type((TransientHTTPError, httpx.TransportError)),
    stop=stop_after_attempt(HTTP.max_retries),
    # Open data APIs enforce quotas over a rolling window, so a 429 often needs to be
    # waited out for a minute or more rather than retried briskly. Capping the backoff at
    # 45s guarantees failure on those; 180s lets a long backfill ride through them.
    wait=wait_exponential_jitter(initial=HTTP.backoff_base_s, max=180.0),
    reraise=True,
)
def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    min_interval_s: float = 0.35,
) -> dict[str, Any]:
    """GET ``url`` and return parsed JSON, retrying transient failures with backoff.

    Parameters
    ----------
    url:
        Absolute request URL.
    params:
        Query parameters. Values that are sequences are joined with commas, which is the
        convention used by both Open-Meteo and the energy-charts API.
    min_interval_s:
        Minimum spacing between consecutive requests to the same host. Open data APIs are
        run on limited budgets; spacing requests keeps us inside their fair-use limits and
        avoids the 429 storms that otherwise appear halfway through a long backfill.
    """
    host = httpx.URL(url).host or "unknown"
    _throttle(host, min_interval_s)

    clean: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if v is None:
            continue
        clean[k] = ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v

    headers = {"User-Agent": HTTP.user_agent, "Accept": "application/json", **HTTP.headers}

    with httpx.Client(timeout=HTTP.timeout_s, follow_redirects=True) as client:
        resp = client.get(url, params=clean, headers=headers)

    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientHTTPError(f"{resp.status_code} from {host}: {resp.text[:200]}")
    resp.raise_for_status()
    return resp.json()
