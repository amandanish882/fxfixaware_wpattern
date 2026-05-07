"""Calibrate turn-of-year overlay magnitudes from observed CME FX futures.

For each day, compute the implied basis between the front-month and the
next-quarter contract:

    basis_bps = (back_close - front_close) / front_close * 10000

Days inside the year-end window (Dec 20-31) are compared against all other
days. The median differential is the empirical turn-of-year widening, in bps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "databento_cache"
    / "fx_curve_history_2024-12-15_to_2026-01-31.parquet"
)

PAIR_FROM_TICKER = {
    "6E": "EUR/USD",
    "6B": "GBP/USD",
    "6J": "JPY/USD",
    "6A": "AUD/USD",
}


def calibrate_toy_overlay(cache_path: "Path | None" = None) -> dict:
    """Return calibrated turn-of-year widening dict ``{pair: bps}``.

    Parameters
    ----------
    cache_path : Path | None
        Path to the cached parquet of CME continuous-contract closes.
        Falls back to the project default cache.

    Returns
    -------
    dict
        Mapping ``{"EUR/USD": widening_bps, ...}``. Empty if the cache is
        missing or has insufficient data.
    """
    path = cache_path if cache_path is not None else CACHE_PATH
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}

    if "ts_event" in df.columns:
        df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 0]).dt.date

    df["ticker_root"] = df["symbol"].str.slice(0, 2)
    df["cont_idx"] = df["symbol"].str.extract(r"\.c\.(\d+)").astype(float)
    df = df.dropna(subset=["cont_idx"])
    df["cont_idx"] = df["cont_idx"].astype(int)

    out: dict = {}
    for ticker, pair in PAIR_FROM_TICKER.items():
        sub = df[df["ticker_root"] == ticker].copy()
        if sub.empty:
            continue
        front = sub[sub["cont_idx"] == 0].set_index("date")["close"]
        back = sub[sub["cont_idx"] == 1].set_index("date")["close"]
        front = front[~front.index.duplicated(keep="last")]
        back = back[~back.index.duplicated(keep="last")]
        merged = pd.DataFrame({"front": front, "back": back}).dropna()
        if merged.empty:
            continue

        merged["basis_bps"] = (
            (merged["back"] - merged["front"]) / merged["front"] * 10000.0
        )
        is_toy = [
            (d.month == 12 and d.day >= 20) for d in merged.index
        ]
        toy_basis = merged.loc[is_toy, "basis_bps"]
        normal_basis = merged.loc[[not x for x in is_toy], "basis_bps"]
        if len(toy_basis) < 2 or len(normal_basis) < 5:
            continue

        widening = float(toy_basis.median() - normal_basis.median())
        out[pair] = round(widening, 1)
    return out
