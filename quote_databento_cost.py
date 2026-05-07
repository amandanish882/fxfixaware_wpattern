"""
Quote-only: ask Databento for the EXACT cost of the planned data fetch.

Calls databento.Historical().metadata.get_cost(...) for each component of the
real-data pipeline, then sums and prints a line-item breakdown.

NO data is fetched. NO billing happens. This is purely a quote API call.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import warnings
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _client():
    import databento as db
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("DATABENTO_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not found in env or .env", file=sys.stderr)
        sys.exit(2)
    return db.Historical(api_key)


def quote(client, label: str, **kwargs) -> Optional[float]:
    try:
        cost = client.metadata.get_cost(**kwargs)
        if isinstance(cost, dict):
            usd = float(cost.get("cost", cost.get("usd", 0.0)))
        else:
            usd = float(cost)
        print(f"  {label:<60} ${usd:>10.4f}")
        return usd
    except Exception as exc:
        print(f"  {label:<60} ERROR: {exc}")
        return None


def main():
    client = _client()

    VAL_DATE = "2026-04-28"
    WIN_START = "2026-01-28"
    FX_TICKERS = ["6E", "6B", "6J", "6A"]

    EXEC_SIM_DAYS = [
        "2026-02-06", "2026-03-05", "2026-03-06", "2026-03-18", "2026-03-19",
        "2026-03-31", "2026-04-03", "2026-04-16", "2026-02-20", "2026-04-22",
    ]

    bdays = pd.bdate_range(WIN_START, VAL_DATE)
    n_bdays = len(bdays)
    fx_cont = [f"{t}.c.0" for t in FX_TICKERS]

    total = 0.0

    print("=" * 78)
    print("DATABENTO COST QUOTE  (metadata.get_cost, no billing)")
    print("=" * 78)
    print(f"Valuation date:      {VAL_DATE}")
    print(f"Fix-pattern window:  {WIN_START} -> {VAL_DATE}  ({n_bdays} trading days)")
    print()

    # -----------------------------------------------------------------
    # 1. MBP-1: quote every business day exactly
    # -----------------------------------------------------------------
    print("--- 1. MBP-1 (top-of-book, 15-17 UTC, 4 FX contracts) PER DAY ---------------")
    mbp1_total = 0.0
    failed = 0
    for d in bdays:
        ds = d.strftime("%Y-%m-%d")
        c = quote(
            client, f"  {ds}",
            dataset="GLBX.MDP3", symbols=fx_cont, schema="mbp-1",
            stype_in="continuous",
            start=f"{ds}T15:00:00", end=f"{ds}T17:00:00",
        )
        if c is not None:
            mbp1_total += c
        else:
            failed += 1
    print(f"  MBP-1 SUBTOTAL ({n_bdays - failed} days, {failed} failed):      ${mbp1_total:>10.4f}")
    total += mbp1_total
    print()

    # -----------------------------------------------------------------
    # 2. MBP-10: every selected day exactly
    # -----------------------------------------------------------------
    print("--- 2. MBP-10 (deep book, 13-17 UTC, 4 FX contracts) PER DAY ----------------")
    mbp10_total = 0.0
    for d in EXEC_SIM_DAYS:
        c = quote(
            client, f"  {d}",
            dataset="GLBX.MDP3", symbols=fx_cont, schema="mbp-10",
            stype_in="continuous",
            start=f"{d}T13:00:00", end=f"{d}T17:00:00",
        )
        if c is not None:
            mbp10_total += c
    print(f"  MBP-10 SUBTOTAL ({len(EXEC_SIM_DAYS)} days):                   ${mbp10_total:>10.4f}")
    total += mbp10_total
    print()

    # -----------------------------------------------------------------
    # 3. SOFR strip (252 EOD)
    # -----------------------------------------------------------------
    print("--- 3. SOFR (SR3) OHLCV-1d strip (USD OIS bootstrap) ------------------------")
    sr3_cont = [f"SR3.c.{i}" for i in range(8)]
    sofr = quote(
        client, "  SR3.c.0..7 OHLCV-1d × 252 days",
        dataset="GLBX.MDP3", symbols=sr3_cont, schema="ohlcv-1d",
        stype_in="continuous",
        start="2025-04-28T00:00:00", end=f"{VAL_DATE}T23:59:59",
    )
    if sofr is not None:
        total += sofr
    print()

    # -----------------------------------------------------------------
    # 4. Back-month FX futures (CCB)
    # -----------------------------------------------------------------
    print("--- 4. Back-month CME FX futures OHLCV-1d (CCB) -----------------------------")
    backcurve = [f"{t}.c.{i}" for t in FX_TICKERS for i in range(4)]
    bm = quote(
        client, "  16 contracts × 1 day (valuation date)",
        dataset="GLBX.MDP3", symbols=backcurve, schema="ohlcv-1d",
        stype_in="continuous",
        start=f"{VAL_DATE}T00:00:00", end=f"{VAL_DATE}T23:59:59",
    )
    if bm is not None:
        total += bm
    print()

    print("=" * 78)
    print(f"  GRAND TOTAL (exact, summed per-day):                       ${total:>10.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
