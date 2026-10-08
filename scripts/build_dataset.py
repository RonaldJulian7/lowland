"""Download every upstream feed and assemble the master hourly panel.

Usage
-----
    python -m scripts.build_dataset                 # incremental, uses caches
    python -m scripts.build_dataset --refresh       # force re-download
    python -m scripts.build_dataset --start 2019-01-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from lowland.config import HISTORY_START
from lowland.dataset import build_panel, congestion_threshold, save_panel
from lowland.io.cache import manifest_table
from lowland.utils import get_logger

log = get_logger("build_dataset")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=HISTORY_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--refresh", action="store_true", help="bypass on-disk caches")
    ap.add_argument("--no-forecast", action="store_true", help="skip the forward NWP window")
    args = ap.parse_args()

    panel = build_panel(
        start=args.start,
        end=args.end,
        refresh=args.refresh,
        include_forecast_window=not args.no_forecast,
    )
    save_panel(panel)

    complete = panel[panel["is_complete"] == 1]
    log.info("complete rows: %s (%.1f%%)", len(complete), 100 * len(complete) / max(len(panel), 1))
    log.info("congestion threshold (P90 residual load): %.0f MW", congestion_threshold(panel))

    pd.set_option("display.width", 160)
    print("\n--- cache manifest ---")
    print(manifest_table().to_string(index=False))

    print("\n--- panel coverage by year ---")
    cov = (
        panel.assign(year=panel.index.year)
        .groupby("year")
        .agg(
            rows=("is_complete", "size"),
            complete=("is_complete", "sum"),
            mean_load_mw=("load", "mean"),
            mean_price=("price_da", "mean"),
            mean_vre_share=("vre_share", "mean"),
        )
        .round(2)
    )
    print(cov.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
