"""Build 15-min cache with both front (.c.0) and back (.c.1) FX futures and
the implied calendar-spread basis.

Reads the bulk ohlcv1m_fx_{start}_to_{end}.parquet file, aggregates 1-minute
closes into 15-minute buckets, pivots so each row is (timestamp, pair) with
both front_close and back_close, and computes:

    log_basis_bps = log(back_close / front_close) * 1e4

Output: data/cme_15m_basis_cache.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "databento_cache"
OUT = PROJECT_ROOT / "data" / "cme_15m_basis_cache.parquet"

PAIR_FROM_TICKER = {"6E": "EUR/USD", "6B": "GBP/USD", "6J": "JPY/USD", "6A": "AUD/USD"}


def main() -> None:
    src = CACHE_DIR / "ohlcv1m_fx_2024-12-01_to_2026-04-28.parquet"
    if not src.exists():
        print(f"[abort] {src} not found — run fetch_ohlcv1m_fx_year.py first")
        return

    print(f"Loading {src} ...")
    raw = pd.read_parquet(src).reset_index()
    print(f"  {len(raw):,} rows")

    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts_event"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df["ticker"] = df["symbol"].str.slice(0, 2)
    cont = df["symbol"].str.extract(r"\.c\.(\d+)")[0]
    df["cont_idx"] = pd.to_numeric(cont, errors="coerce")
    df = df.dropna(subset=["cont_idx", "close"])
    df["cont_idx"] = df["cont_idx"].astype(int)
    df = df[df["cont_idx"].isin([0, 1])]
    df = df[df["ticker"].isin(PAIR_FROM_TICKER)]
    df["pair"] = df["ticker"].map(PAIR_FROM_TICKER)
    df = df[["ts", "pair", "cont_idx", "close"]]
    print(f"  filtered: {len(df):,} rows, {df['pair'].nunique()} pairs, "
          f"{df['ts'].dt.date.nunique()} days")

    # 15-min buckets: take last 1m close per bucket
    df["bucket"] = df["ts"].dt.floor("15min")
    last = (
        df.sort_values("ts")
          .groupby(["bucket", "pair", "cont_idx"], as_index=False)
          .agg(close=("close", "last"))
    )

    wide = last.pivot_table(
        index=["bucket", "pair"], columns="cont_idx",
        values="close", aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={0: "front_close", 1: "back_close"})
    wide = wide.dropna(subset=["front_close", "back_close"])

    wide["log_basis_bps"] = np.log(wide["back_close"] / wide["front_close"]) * 1e4

    wide = wide.sort_values(["pair", "bucket"]).reset_index(drop=True)

    # Session boundary: define session breaks at gaps > 4 hours
    wide["session"] = wide.groupby("pair")["bucket"].transform(
        lambda x: (x.diff() > pd.Timedelta(hours=4)).cumsum()
    )

    # Per-session 15m log returns for front and back
    wide["front_ret_15m"] = wide.groupby(["pair", "session"])["front_close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    wide["back_ret_15m"] = wide.groupby(["pair", "session"])["back_close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    # Basis change per 15m
    wide["basis_change_bps"] = wide.groupby(["pair", "session"])["log_basis_bps"].transform(
        lambda x: x.diff()
    )

    wide = wide.rename(columns={"bucket": "timestamp"})
    out_cols = ["timestamp", "pair", "front_close", "back_close",
                "log_basis_bps", "front_ret_15m", "back_ret_15m",
                "basis_change_bps"]
    wide = wide[out_cols]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print(f"  rows={len(wide):,}  date range="
          f"{wide['timestamp'].min().date()} → {wide['timestamp'].max().date()}")
    print()
    print(wide.groupby("pair").agg(
        n_buckets=("timestamp", "size"),
        n_days=("timestamp", lambda x: x.dt.date.nunique()),
        mean_basis_bps=("log_basis_bps", "mean"),
        std_basis_bps=("log_basis_bps", "std"),
    ).round(2))


if __name__ == "__main__":
    main()
