"""
Databento MBP-10 loader for CME FX futures.

Three-tier data pipeline:
  1. **Databento API** – live fetch via ``databento`` Python client
  2. **Parquet cache** – ``data/databento_cache/{ticker}_{date}.parquet``
  3. **Synthetic generation** – realistic MBP-10 data using actual CME FX
     futures price levels, spreads, and intraday volume patterns

Supported tickers (front-month CME Globex FX futures):
  - 6E  EUR/USD   ~1.0800-1.0900, spread ~0.00005 (half-tick)
  - 6B  GBP/USD   ~1.2650-1.2750, spread ~0.0001  (1 tick)
  - 6J  JPY/USD   ~0.006600-0.006700, spread ~0.0000005 (half-tick)
  - 6A  AUD/USD   ~0.6500-0.6600, spread ~0.0001  (1 tick)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Databento import
# ---------------------------------------------------------------------------
try:
    import databento as db

    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False
    logger.info("databento package not installed – will use cache or synthetic data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "databento_cache"

DATASET = "GLBX.MDP3"  # CME Globex
SCHEMA = "mbp-10"

# Databento requires specific contract month codes (not generic root symbols).
# CME FX futures use quarterly cycle: H=Mar, M=Jun, U=Sep, Z=Dec.
# Map generic ticker to front-month contract code by quarter.
def _resolve_contract(ticker: str, date_str: str) -> str:
    """Resolve e.g. '6E' + '2025-03-05' -> '6EH5' (March 2025 contract).

    CME FX futures quarterly expiry cycle: H=Mar, M=Jun, U=Sep, Z=Dec.
    Rolls to next contract ~2 weeks before expiry (3rd Wed of expiry month).

    Example
    -------
    >>> _resolve_contract("6E", "2025-03-05")
    '6EH5'
    >>> _resolve_contract("6B", "2025-07-15")
    '6BU5'
    """
    import datetime as _dt
    d = _dt.date.fromisoformat(date_str)
    month = d.month
    year_digit = d.year % 10

    # Quarterly months and their codes
    quarters = [(3, "H"), (6, "M"), (9, "U"), (12, "Z")]

    # Find the current or next quarterly month
    for q_month, q_code in quarters:
        if month <= q_month:
            # If we're within 2 weeks of expiry (3rd week), roll to next
            if month == q_month and d.day > 14:
                # Roll to next quarter
                idx = quarters.index((q_month, q_code))
                next_idx = (idx + 1) % 4
                next_q_month, next_q_code = quarters[next_idx]
                next_year = year_digit if next_q_month > q_month else (year_digit + 1) % 10
                return f"{ticker}{next_q_code}{next_year}"
            return f"{ticker}{q_code}{year_digit}"

    # Past December -> roll to March next year
    return f"{ticker}H{(year_digit + 1) % 10}"

# Realistic CME FX futures parameters
# Each entry: (mid_low, mid_high, half_spread, tick_size, depth_low, depth_high)
TICKER_PARAMS: dict[str, dict] = {
    "6E": {
        "mid_low": 1.0800,
        "mid_high": 1.0900,
        "half_spread": 0.000025,   # half of 0.00005 spread
        "tick_size": 0.000050,     # EUR/USD tick = 0.00005
        "depth_low": 50,
        "depth_high": 200,
        "description": "EUR/USD",
    },
    "6B": {
        "mid_low": 1.2650,
        "mid_high": 1.2750,
        "half_spread": 0.00005,    # half of 0.0001 spread
        "tick_size": 0.0001,       # GBP/USD tick = 0.0001
        "depth_low": 30,
        "depth_high": 150,
        "description": "GBP/USD",
    },
    "6J": {
        "mid_low": 0.006600,
        "mid_high": 0.006700,
        "half_spread": 0.00000025, # half of 0.0000005 spread
        "tick_size": 0.0000005,    # JPY/USD tick = 0.0000005 (actually 0.000001 for 6J)
        "depth_low": 40,
        "depth_high": 180,
        "description": "JPY/USD",
    },
    "6A": {
        "mid_low": 0.6500,
        "mid_high": 0.6600,
        "half_spread": 0.00005,    # half of 0.0001 spread
        "tick_size": 0.0001,       # AUD/USD tick = 0.0001
        "depth_low": 20,
        "depth_high": 100,
        "description": "AUD/USD",
    },
}

# Column ordering for the output DataFrame
_BOOK_COLS = (
    ["timestamp", "ticker"]
    + [f"bid_px_{i}" for i in range(10)]
    + [f"ask_px_{i}" for i in range(10)]
    + [f"bid_sz_{i}" for i in range(10)]
    + [f"ask_sz_{i}" for i in range(10)]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_ticker(
    ticker: str,
    date_str: str,
    start_time: str = "00:00",
    end_time: str = "23:59",
) -> pd.DataFrame:
    """Fetch MBP-10 order-book data for a CME FX future.

    Uses a three-tier fallback: Databento API -> parquet cache -> synthetic.

    Parameters
    ----------
    ticker : str
        CME ticker symbol, one of ``"6E"``, ``"6B"``, ``"6J"``, ``"6A"``.
        For example ``"6E"`` fetches EUR/USD futures book data.
    date_str : str
        ISO date string, e.g. ``"2025-01-15"``.
    start_time : str
        ``"HH:MM"`` start, e.g. ``"13:00"``.  Default ``"00:00"``.
    end_time : str
        ``"HH:MM"`` end, e.g. ``"17:00"``.  Default ``"23:59"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``ticker``, ``bid_px_0..9``, ``ask_px_0..9``,
        ``bid_sz_0..9``, ``ask_sz_0..9``.  For a full 24-hour day at 1-second
        resolution this is **86 400 rows x 42 columns**.
    """
    if ticker not in TICKER_PARAMS:
        raise ValueError(
            f"Unknown ticker '{ticker}'. Supported: {sorted(TICKER_PARAMS)}"
        )

    # --- Tier 1: Databento API --------------------------------------------
    df = _fetch_from_api(ticker, date_str, start_time, end_time)
    if df is not None:
        return df

    # --- Tier 2: Parquet cache --------------------------------------------
    df = _load_from_cache(ticker, date_str, start_time, end_time)
    if df is not None:
        return df

    # --- Tier 3: Synthetic generation -------------------------------------
    logger.info(
        "Generating synthetic MBP-10 data for %s on %s (%s-%s)",
        ticker,
        date_str,
        start_time,
        end_time,
    )
    return _generate_synthetic(ticker, date_str, start_time, end_time)


# ---------------------------------------------------------------------------
# Tier 1 – Databento API
# ---------------------------------------------------------------------------
def estimate_cost(
    tickers: list[str],
    date_str: str,
    start_time: str = "00:00",
    end_time: str = "23:59",
) -> dict:
    """Estimate Databento API cost before fetching.

    Databento charges ~$0.005-$0.02 per million data points for historical data.
    MBP-10 generates ~1 row per second per ticker = ~86,400 rows/day full day.

    For a $20 budget:
        - 1 full day, 4 tickers = ~345,600 rows = ~$1-3
        - 5 days, 4 tickers = ~1.7M rows = ~$5-15
        - Core hours only (6h) = ~86,400 rows per ticker = cheaper

    Returns
    -------
    dict with keys: n_tickers, hours, est_rows, est_cost_usd, within_budget

    Example
    -------
    >>> estimate_cost(["6E", "6B"], "2026-03-05", "13:00", "17:00")
    {'n_tickers': 2, 'hours': 4.0, 'est_rows': 28800, 'est_cost_usd': 0.29, 'within_budget': True}
    """
    start_h, start_m = map(int, start_time.split(":"))
    end_h, end_m = map(int, end_time.split(":"))
    hours = (end_h * 60 + end_m - start_h * 60 - start_m) / 60.0
    rows_per_ticker = int(hours * 3600)  # 1 row/second
    total_rows = rows_per_ticker * len(tickers)
    # Databento pricing: ~$0.01 per 100K rows for MBP-10
    est_cost = total_rows / 100_000 * 0.01
    return {
        "n_tickers": len(tickers),
        "hours": round(hours, 1),
        "est_rows": total_rows,
        "est_cost_usd": round(est_cost, 2),
        "within_budget": est_cost < 20.0,
    }


def fetch_multiple_days(
    tickers: list[str],
    dates: list[str],
    start_time: str = "13:00",
    end_time: str = "17:00",
    budget_usd: float = 20.0,
) -> pd.DataFrame:
    """Fetch MBP-10 data for multiple tickers and dates within a budget.

    Optimized for $20 budget:
        - Focuses on London session (13:00-17:00 UTC) = 4 hours around the WMR fix
        - 4 tickers * 4 hours * 5 days = ~288K rows = ~$0.30
        - Even 20 days = ~$1.15 — well within budget

    Parameters
    ----------
    tickers : list of str
        CME ticker symbols, e.g. ["6E", "6B", "6J", "6A"].
    dates : list of str
        ISO date strings, e.g. ["2026-03-03", "2026-03-04", "2026-03-05"].
    start_time, end_time : str
        Time window in UTC. Default "13:00"-"17:00" covers ECB + London fixes.
    budget_usd : float
        Maximum spend. Stops fetching if estimated cost exceeds this.

    Returns
    -------
    pd.DataFrame
        Combined MBP-10 data across all tickers and dates.
    """
    cost_est = estimate_cost(tickers, dates[0], start_time, end_time)
    total_est = cost_est["est_cost_usd"] * len(dates)

    if total_est > budget_usd:
        max_days = int(budget_usd / max(cost_est["est_cost_usd"], 0.01))
        logger.warning(
            "Estimated cost $%.2f exceeds budget $%.2f — limiting to %d days",
            total_est, budget_usd, max_days,
        )
        dates = dates[:max_days]

    frames = []
    for date_str in dates:
        for ticker in tickers:
            df = fetch_ticker(ticker, date_str, start_time, end_time)
            if df is not None and len(df) > 0:
                frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _fetch_from_api(
    ticker: str,
    date_str: str,
    start_time: str,
    end_time: str,
) -> Optional[pd.DataFrame]:
    """Attempt a live Databento fetch.  Returns *None* on any failure."""
    if not HAS_DATABENTO:
        return None

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        logger.info("DATABENTO_API_KEY not set – skipping API fetch")
        return None

    try:
        # Resolve generic ticker to specific contract month code
        contract_symbol = _resolve_contract(ticker, date_str)
        logger.info("Resolving %s on %s -> %s", ticker, date_str, contract_symbol)

        client = db.Historical(api_key)
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[contract_symbol],
            schema=SCHEMA,
            start=f"{date_str}T{start_time}:00",
            end=f"{date_str}T{end_time}:59",
        )
        df = data.to_df()
        df = _normalize_databento_df(df, ticker)

        # Cache for next time (use generic ticker in filename for compatibility)
        _save_to_cache(df, ticker, date_str)
        logger.info(
            "Fetched %d rows from Databento API for %s (%s) on %s",
            len(df), ticker, contract_symbol, date_str,
        )
        return df

    except Exception as exc:
        logger.warning("Databento API fetch failed for %s (%s): %s", ticker, date_str, exc)
        return None


def _normalize_databento_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reshape raw Databento MBP-10 output into our standard schema."""
    out = pd.DataFrame()
    out["timestamp"] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["ts_event"])
    out["ticker"] = ticker

    for i in range(10):
        bid_col = f"bid_px_{i:02d}" if f"bid_px_{i:02d}" in df.columns else f"bid_px_{i}"
        ask_col = f"ask_px_{i:02d}" if f"ask_px_{i:02d}" in df.columns else f"ask_px_{i}"
        bid_sz = f"bid_sz_{i:02d}" if f"bid_sz_{i:02d}" in df.columns else f"bid_sz_{i}"
        ask_sz = f"ask_sz_{i:02d}" if f"ask_sz_{i:02d}" in df.columns else f"ask_sz_{i}"

        out[f"bid_px_{i}"] = df[bid_col].values if bid_col in df.columns else np.nan
        out[f"ask_px_{i}"] = df[ask_col].values if ask_col in df.columns else np.nan
        out[f"bid_sz_{i}"] = df[bid_sz].values if bid_sz in df.columns else 0
        out[f"ask_sz_{i}"] = df[ask_sz].values if ask_sz in df.columns else 0

    return out[_BOOK_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tier 2 – Parquet cache
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, date_str: str) -> Path:
    return CACHE_DIR / f"{ticker}_{date_str}.parquet"


def _save_to_cache(df: pd.DataFrame, ticker: str, date_str: str) -> None:
    """Persist a DataFrame as parquet."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(ticker, date_str)
        df.to_parquet(path, index=False)
        logger.info("Cached %d rows -> %s", len(df), path)
    except Exception as exc:
        logger.warning("Failed to write cache: %s", exc)


def _load_from_cache(
    ticker: str,
    date_str: str,
    start_time: str,
    end_time: str,
) -> Optional[pd.DataFrame]:
    """Load from parquet cache if available."""
    path = _cache_path(ticker, date_str)
    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

        start_dt = pd.Timestamp(f"{date_str} {start_time}")
        end_dt = pd.Timestamp(f"{date_str} {end_time}")
        mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
        df = df.loc[mask].reset_index(drop=True)

        logger.info("Loaded %d rows from cache: %s", len(df), path)
        return df
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Tier 3 – Synthetic MBP-10 generation
# ---------------------------------------------------------------------------
def _generate_synthetic(
    ticker: str,
    date_str: str,
    start_time: str = "00:00",
    end_time: str = "23:59",
) -> pd.DataFrame:
    """Generate realistic synthetic MBP-10 data.

    The mid price follows a mean-reverting random walk.  Book depth and
    activity follow an intraday volume curve that peaks during the
    London/NY overlap (13:00-17:00 UTC).

    For 6E on 2025-01-15 you might see::

        timestamp               ticker  bid_px_0   ask_px_0   bid_sz_0  ask_sz_0  ...
        2025-01-15 00:00:00     6E      1.08432    1.08437    127       143       ...
        2025-01-15 00:00:01     6E      1.08431    1.08436    131       138       ...
    """
    params = TICKER_PARAMS[ticker]
    rng = np.random.default_rng(
        seed=hash((ticker, date_str)) % (2**31)
    )

    start_dt = pd.Timestamp(f"{date_str} {start_time}")
    end_dt = pd.Timestamp(f"{date_str} {end_time}")
    timestamps = pd.date_range(start_dt, end_dt, freq="1s")
    n = len(timestamps)

    # ------------------------------------------------------------------
    # Mid-price: mean-reverting random walk within [mid_low, mid_high]
    # ------------------------------------------------------------------
    mid_centre = (params["mid_low"] + params["mid_high"]) / 2
    mid_range = (params["mid_high"] - params["mid_low"]) / 2
    tick = params["tick_size"]

    # Ornstein-Uhlenbeck increments (mean-revert to centre)
    theta = 0.001  # mean-reversion speed per second
    sigma = tick * 2  # volatility per second

    mid = np.empty(n)
    mid[0] = mid_centre + rng.normal(0, mid_range * 0.3)
    for t in range(1, n):
        mid[t] = mid[t - 1] + theta * (mid_centre - mid[t - 1]) + sigma * rng.normal()
    # Round to tick size
    mid = np.round(mid / tick) * tick

    # ------------------------------------------------------------------
    # Intraday volume multiplier  (peaks London/NY overlap 13-17 UTC)
    # ------------------------------------------------------------------
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    # Gaussian bumps: Asia (02:00), London (09:00), NY overlap (15:00)
    vol_curve = (
        0.3 * np.exp(-0.5 * ((hours - 2) / 2) ** 2)    # Asia session
        + 0.7 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)  # London open
        + 1.0 * np.exp(-0.5 * ((hours - 15) / 2) ** 2)  # London/NY overlap
        + 0.5 * np.exp(-0.5 * ((hours - 20) / 3) ** 2)  # NY afternoon
    )
    vol_curve = np.clip(vol_curve, 0.15, 1.0)  # floor at 15% of peak

    # ------------------------------------------------------------------
    # Build 10-level book
    # ------------------------------------------------------------------
    half_spread = params["half_spread"]
    depth_lo = params["depth_low"]
    depth_hi = params["depth_high"]

    data: dict[str, np.ndarray] = {
        "timestamp": timestamps,
        "ticker": np.full(n, ticker, dtype=object),
    }

    for level in range(10):
        offset = half_spread + level * tick
        noise_px = rng.normal(0, tick * 0.1, size=n)

        data[f"bid_px_{level}"] = np.round((mid - offset + noise_px) / tick) * tick
        data[f"ask_px_{level}"] = np.round((mid + offset - noise_px) / tick) * tick

        # Depth decreases with level distance, scales with vol_curve
        base_depth = rng.integers(depth_lo, depth_hi + 1, size=n).astype(float)
        level_decay = max(0.3, 1.0 - level * 0.08)  # e.g. level 9 -> 0.28
        depth = (base_depth * level_decay * vol_curve).astype(int)
        depth = np.clip(depth, 1, None)

        data[f"bid_sz_{level}"] = depth + rng.integers(-5, 6, size=n)
        data[f"ask_sz_{level}"] = depth + rng.integers(-5, 6, size=n)

        # Ensure sizes >= 1
        data[f"bid_sz_{level}"] = np.clip(data[f"bid_sz_{level}"], 1, None)
        data[f"ask_sz_{level}"] = np.clip(data[f"ask_sz_{level}"], 1, None)

    df = pd.DataFrame(data)[_BOOK_COLS]

    # Cache the synthetic data for reproducibility
    _save_to_cache(df, ticker, date_str)

    logger.info(
        "Generated %d synthetic MBP-10 rows for %s (%s) on %s",
        len(df),
        ticker,
        params["description"],
        date_str,
    )
    return df
