"""Fetch OHLCV-1m for 6E/6B/6J/6A, both .c.0 (front) and .c.1 (back-quarter),
across the full date range 2024-12-01 to 2026-04-28 (one bulk request).

Saves a single combined parquet at:
    data/databento_cache/ohlcv1m_fx_2024-12-01_to_2026-04-28.parquet

Quoted cost on 2026-03-31 (full day, 8 symbols, ohlcv-1m): $0.00181/day,
so ~370 days ≈ $0.70 total. Safety threshold is $5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefetch_data import _client

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "databento_cache"

DATASET = "GLBX.MDP3"
TICKERS = ["6E", "6B", "6J", "6A"]
SYMBOLS = [f"{t}.c.0" for t in TICKERS] + [f"{t}.c.1" for t in TICKERS]
START = "2024-12-01"
END = "2026-04-28"


def main() -> None:
    client = _client()
    if client is None:
        print("[abort] No DATABENTO_API_KEY")
        sys.exit(1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out = CACHE_DIR / f"ohlcv1m_fx_{START}_to_{END}.parquet"
    if out.exists():
        print(f"already cached at {out}")
        df = pd.read_parquet(out)
        print(f"  rows={len(df):,}, date range="
              f"{pd.to_datetime(df['ts_event']).min()} → "
              f"{pd.to_datetime(df['ts_event']).max()}")
        return

    print(f"Cost-quote: {START} → {END}, 8 symbols, ohlcv-1m")
    cost = client.metadata.get_cost(
        dataset=DATASET, symbols=SYMBOLS, schema="ohlcv-1m",
        stype_in="continuous",
        start=f"{START}T00:00:00", end=f"{END}T23:59:59",
    )
    c = float(cost.get("cost", cost.get("usd", 0.0))) if isinstance(cost, dict) else float(cost)
    print(f"  total cost: ${c:.4f}")
    if c > 9.0:
        print(f"[abort] cost exceeds $9 safety threshold")
        sys.exit(2)

    print(f"\nFetching... (single bulk request for the full date range)")
    data = client.timeseries.get_range(
        dataset=DATASET, symbols=SYMBOLS, schema="ohlcv-1m",
        stype_in="continuous",
        start=f"{START}T00:00:00", end=f"{END}T23:59:59",
    )
    df = data.to_df()
    print(f"  received: {len(df):,} rows")
    if df is None or df.empty:
        print("[abort] empty response")
        sys.exit(3)

    df.to_parquet(out)
    print(f"  saved: {out}")

    # Summary
    df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    print(f"\n  date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  unique dates: {df['date'].nunique()}")
    print(f"  symbols seen: {sorted(df['symbol'].unique())}")
    print(f"  rows per symbol:")
    print(df.groupby("symbol").size().to_string())


if __name__ == "__main__":
    main()
