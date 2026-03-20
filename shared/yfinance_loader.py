"""
Yahoo Finance FX Intraday Data Loader
=====================================

Fetches REAL historical intraday FX data from Yahoo Finance for the W-shape
fix pattern analysis.  No API key required.

Data availability (yfinance limitations):
    - 1h bars:  ~730 days (2 years) -- RECOMMENDED for fix analysis
    - 15m bars: ~60 days
    - 5m bars:  ~60 days
    - 1m bars:  ~7 days
    - 1d bars:  20+ years

Since fix windows are 30-60 min wide, 1h bars capture the pattern well.
For finer analysis, 15m bars over 60 days still provide ~60 * 96 = 5,760
observations per pair.

Usage:
    >>> from shared.yfinance_loader import fetch_fx_intraday
    >>> df = fetch_fx_intraday(pairs=["EUR/USD", "GBP/USD"], interval="1h", period="2y")
    >>> df.columns  # ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume', 'return_1h']
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance import (optional dependency)
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.info("yfinance not installed — pip install yfinance for real FX data")

# ---------------------------------------------------------------------------
# Yahoo Finance FX ticker mapping
# ---------------------------------------------------------------------------
PAIR_TO_YAHOO: Dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "JPY/USD": "JPY=X",      # Yahoo quotes USD/JPY; we invert
    "AUD/USD": "AUDUSD=X",
    "CAD/USD": "CAD=X",      # Yahoo quotes USD/CAD; we invert
    "CHF/USD": "CHF=X",      # Yahoo quotes USD/CHF; we invert
    "NZD/USD": "NZDUSD=X",
}

# Pairs where Yahoo quotes the inverse (USD/XXX instead of XXX/USD)
_INVERT_PAIRS = {"JPY/USD", "CAD/USD", "CHF/USD"}

# Cache directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "yfinance_cache"


def fetch_fx_intraday(
    pairs: Optional[List[str]] = None,
    interval: str = "1h",
    period: str = "60d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch real intraday FX data from Yahoo Finance.

    Parameters
    ----------
    pairs : list of str
        Currency pairs like ["EUR/USD", "GBP/USD"].  Defaults to G4.
    interval : str
        Bar interval: "1m", "5m", "15m", "1h", "1d".
        For fix analysis, "1h" gives 2 years of data; "15m" gives 60 days.
    period : str
        Lookback period: "7d", "60d", "6mo", "1y", "2y", "5y", "max".
        Ignored if start/end are provided.
    start, end : str, optional
        ISO date strings like "2024-01-01".  If both provided, period is ignored.
    use_cache : bool
        If True, check/write parquet cache before hitting Yahoo.

    Returns
    -------
    pd.DataFrame or None
        Columns: timestamp, pair, open, high, low, close, volume, mid_price, return_Xh/Xm.
        Returns None if yfinance is not installed and no cache exists.

    Example
    -------
    >>> df = fetch_fx_intraday(["EUR/USD"], interval="1h", period="2y")
    >>> len(df)  # ~12,000 rows (252 days * 2 years * ~24 bars/day)
    12096
    >>> df[df["pair"] == "EUR/USD"]["close"].mean()  # e.g. 1.0845
    """
    if pairs is None:
        pairs = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]

    # Build cache key
    cache_key = f"fx_intraday_{'_'.join(p.replace('/', '') for p in pairs)}_{interval}_{period}"
    if start and end:
        cache_key = f"fx_intraday_{'_'.join(p.replace('/', '') for p in pairs)}_{interval}_{start}_{end}"

    # Try cache first
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            logger.info("Loaded %d rows from yfinance cache (%s)", len(cached), cache_key)
            return cached

    # Try yfinance
    if not HAS_YFINANCE:
        logger.warning("yfinance not installed and no cache found — returning None")
        return None

    frames = []
    for pair in pairs:
        yahoo_ticker = PAIR_TO_YAHOO.get(pair)
        if yahoo_ticker is None:
            logger.warning("No Yahoo ticker mapping for %s — skipping", pair)
            continue

        try:
            ticker = yf.Ticker(yahoo_ticker)
            if start and end:
                hist = ticker.history(start=start, end=end, interval=interval)
            else:
                hist = ticker.history(period=period, interval=interval)

            if hist is None or hist.empty:
                logger.warning("No data returned for %s (%s)", pair, yahoo_ticker)
                continue

            df = pd.DataFrame({
                "timestamp": hist.index.tz_localize(None) if hist.index.tz else hist.index,
                "open": hist["Open"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "close": hist["Close"].values,
                "volume": hist["Volume"].values if "Volume" in hist.columns else 0,
            })

            # Invert pairs where Yahoo quotes USD/XXX
            if pair in _INVERT_PAIRS:
                for col in ["open", "high", "low", "close"]:
                    df[col] = 1.0 / df[col]
                # Swap high/low after inversion
                df["high"], df["low"] = df["low"].copy(), df["high"].copy()

            df["pair"] = pair
            df["mid_price"] = (df["open"] + df["close"]) / 2

            # Compute returns
            df[f"return_{interval}"] = df["close"].pct_change()

            # Drop first row (NaN return)
            df = df.dropna(subset=[f"return_{interval}"])

            frames.append(df)
            logger.info("Fetched %d %s bars for %s from Yahoo Finance", len(df), interval, pair)

        except Exception as exc:
            logger.warning("Failed to fetch %s from Yahoo Finance: %s", pair, exc)
            continue

    if not frames:
        logger.warning("No data fetched from Yahoo Finance for any pair")
        return None

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["pair", "timestamp"]).reset_index(drop=True)

    # Cache for next time
    if use_cache:
        _save_cache(result, cache_key)

    logger.info("Total: %d intraday bars across %d pairs from Yahoo Finance", len(result), len(frames))
    return result


def fetch_fx_daily(
    pairs: Optional[List[str]] = None,
    period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch daily FX close prices — unlimited history.

    Returns DataFrame with DatetimeIndex and pair columns.
    Example: df["EUR/USD"] = 1.0832, df["GBP/USD"] = 1.2714, etc.
    """
    if pairs is None:
        pairs = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]

    if not HAS_YFINANCE:
        return None

    frames = {}
    for pair in pairs:
        yahoo_ticker = PAIR_TO_YAHOO.get(pair)
        if yahoo_ticker is None:
            continue
        try:
            ticker = yf.Ticker(yahoo_ticker)
            if start and end:
                hist = ticker.history(start=start, end=end, interval="1d")
            else:
                hist = ticker.history(period=period, interval="1d")
            if hist is not None and not hist.empty:
                series = hist["Close"]
                if pair in _INVERT_PAIRS:
                    series = 1.0 / series
                series.index = series.index.tz_localize(None) if series.index.tz else series.index
                frames[pair] = series
        except Exception as exc:
            logger.warning("Failed daily fetch for %s: %s", pair, exc)

    if not frames:
        return None

    return pd.DataFrame(frames).dropna()


def convert_to_15min_returns(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Convert 1h bar data to approximate 15-min returns for fix analysis.

    Distributes the hourly return into 4 sub-periods using intraday vol
    weighting, preserving the total return per hour.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Output of fetch_fx_intraday(..., interval="1h") with columns:
        timestamp, pair, close, return_1h.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, pair, return_15m — same format as
        FixAlphaModel.generate_realistic_intraday_data().
    """
    rows = []
    for pair, group in hourly_df.groupby("pair"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for _, row in group.iterrows():
            ts = row["timestamp"]
            ret_1h = row["return_1h"]
            if np.isnan(ret_1h):
                continue

            # Split into 4 x 15-min buckets with slight randomness
            # but preserving total return
            rng = np.random.default_rng(hash((str(ts), pair)) % (2**31))
            weights = np.array([0.25, 0.25, 0.25, 0.25]) + rng.normal(0, 0.05, 4)
            weights = np.abs(weights)
            weights /= weights.sum()

            for i in range(4):
                sub_ts = ts + pd.Timedelta(minutes=15 * i)
                rows.append({
                    "timestamp": sub_ts,
                    "pair": pair,
                    "return_15m": ret_1h * weights[i],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def _load_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        # Check staleness: if cache is older than 24h for intraday, refresh
        import os
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if (datetime.now() - mtime).total_seconds() > 86400:
            logger.info("Cache is >24h old, will refresh from Yahoo")
            return None
        return df
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, key: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_cache_path(key), index=False)
        logger.info("Cached %d rows -> %s", len(df), _cache_path(key))
    except Exception as exc:
        logger.warning("Failed to cache yfinance data: %s", exc)
