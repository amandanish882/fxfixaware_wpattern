"""
W-Shaped FX Fix Pattern Alpha Signals
======================================

Implements the core finding from Krohn, Mueller & Whelan (2024, Journal of Finance):
USD systematically *appreciates* before each of the three daily FX fixes, then
*depreciates* after. This creates a "W" shape in intraday cumulative USD returns.

The three daily FX fixes (UTC times):
    1. Tokyo fix:   00:55 UTC  (9:55 JST)
    2. ECB fix:     13:15 UTC  (14:15 CET)
    3. London WMR:  16:00 UTC  (4:00 PM GMT) -- the dominant fix

Key empirical facts:
    - Pre-fix USD appreciation: ~1-3 bps per fix window across all G4 pairs
    - Post-fix reversal: ~0.5-2 bps in the opposite direction
    - Annualized Sharpe of fix-timing strategy: ~2-5 (purely from timing)
    - Effect is strongest for the WMR 4PM London fix
    - Driven by corporate and asset manager hedging demand (predictable USD buying)
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Standard FX fix times (UTC)
# ======================================================================
DEFAULT_FIX_TIMES = {
    "tokyo":  time(0, 55),    # 00:55 UTC = 9:55 JST
    "ecb":    time(13, 15),   # 13:15 UTC = 14:15 CET
    "london": time(16, 0),    # 16:00 UTC = 4:00 PM GMT (WMR fix)
}

# Pairs and their typical daily vol in pips (for simulation)
PAIR_DAILY_VOL_PIPS = {
    "EUR/USD": 50,
    "GBP/USD": 70,
    "JPY/USD": 60,
    "AUD/USD": 80,
}

# 15-min bucket vol as fraction of daily vol (intraday vol pattern)
# Low in Asia, peaks during London/NY overlap, tapers in late NY
INTRADAY_VOL_PATTERN_96 = None  # Lazy-initialized


def _build_intraday_vol_pattern() -> np.ndarray:
    """Build a 96-element array (one per 15-min bucket) of relative vol weights.

    The pattern peaks during London-NY overlap (13:00-17:00 UTC) and is
    lowest during the late-Asia / early-Europe gap.

    Example values:
        Bucket 0  (00:00 UTC): 0.40  (quiet Asia)
        Bucket 52 (13:00 UTC): 1.80  (London open + NY open overlap)
        Bucket 64 (16:00 UTC): 1.60  (WMR fix time)
        Bucket 80 (20:00 UTC): 0.60  (late NY)
    """
    pattern = np.ones(96)
    for i in range(96):
        hour = (i * 15) / 60.0  # UTC hour as float, e.g. bucket 52 -> 13.0
        if hour < 3:
            pattern[i] = 0.40  # Late Asia
        elif hour < 7:
            pattern[i] = 0.55  # Early Europe
        elif hour < 8:
            pattern[i] = 0.90  # Frankfurt/London pre-open
        elif hour < 12:
            pattern[i] = 1.20  # London morning
        elif hour < 13:
            pattern[i] = 1.50  # Pre-NY open
        elif hour < 17:
            pattern[i] = 1.80  # London-NY overlap (peak)
        elif hour < 18:
            pattern[i] = 1.30  # Post-London close
        elif hour < 21:
            pattern[i] = 0.70  # NY afternoon
        else:
            pattern[i] = 0.45  # Late NY / early Asia
    # Normalize so they sum to 96 (i.e. average weight = 1.0)
    pattern = pattern / pattern.mean()
    return pattern


# ======================================================================
# CME MBP-1 / MBP-10 -> 15-min log-return aggregation
# ======================================================================

DEFAULT_CME_TICKER_TO_PAIR = {
    "6E": "EUR/USD",
    "6B": "GBP/USD",
    "6J": "JPY/USD",
    "6A": "AUD/USD",
}


def aggregate_cme_to_15m_returns(
    mbp_df: pd.DataFrame,
    ticker_to_pair: dict,
) -> pd.DataFrame:
    """Aggregate a CME MBP-1 / MBP-10 stream into 15-min mid-price log returns.

    Parameters
    ----------
    mbp_df : pd.DataFrame
        Must contain ``ts_event`` (or ``timestamp``), ``symbol`` (or ``ticker``),
        plus top-of-book bid/ask. Bid/ask column names are auto-detected:
        ``bid_px_00``/``ask_px_00`` first, then ``bid_px_0``/``ask_px_0``.
    ticker_to_pair : dict
        e.g. ``{"6E": "EUR/USD", "6B": "GBP/USD", "6J": "JPY/USD", "6A": "AUD/USD"}``.

    Returns
    -------
    pd.DataFrame
        Columns ``timestamp``, ``pair``, ``return_15m`` (log returns).
    """
    empty = pd.DataFrame(columns=["timestamp", "pair", "return_15m"])

    if mbp_df is None or len(mbp_df) == 0:
        return empty

    df = mbp_df

    # 1) Timestamp column
    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True)
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, utc=True)
    else:
        return empty

    # 2) Bid/ask columns
    if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
        bid = df["bid_px_00"].astype(float).to_numpy()
        ask = df["ask_px_00"].astype(float).to_numpy()
    elif "bid_px_0" in df.columns and "ask_px_0" in df.columns:
        bid = df["bid_px_0"].astype(float).to_numpy()
        ask = df["ask_px_0"].astype(float).to_numpy()
    else:
        return empty

    # 3) Symbol / ticker column
    if "symbol" in df.columns:
        sym = df["symbol"].astype(str).to_numpy()
    elif "ticker" in df.columns:
        sym = df["ticker"].astype(str).to_numpy()
    else:
        return empty

    mid = (bid + ask) / 2.0

    # 4) Map each row to a ticker root via prefix-match
    roots_arr = np.full(len(sym), None, dtype=object)
    ticker_keys = list(ticker_to_pair.keys())
    for i, s in enumerate(sym):
        for root in ticker_keys:
            if s.startswith(root):
                roots_arr[i] = root
                break

    work = pd.DataFrame({
        "ts": ts.values if hasattr(ts, "values") else ts,
        "mid": mid,
        "root": roots_arr,
    })
    work = work.dropna(subset=["root", "mid"])
    if work.empty:
        return empty

    out_frames = []
    for root, group in work.groupby("root"):
        pair = ticker_to_pair.get(root)
        if pair is None:
            continue
        g = group.set_index("ts").sort_index()
        bucket = g["mid"].resample("15min", label="left", closed="left").last().dropna()
        if len(bucket) < 2:
            continue
        rets = np.log(bucket).diff().dropna()
        if rets.empty:
            continue
        out_frames.append(pd.DataFrame({
            "timestamp": rets.index,
            "pair": pair,
            "return_15m": rets.values,
        }))

    if not out_frames:
        return empty

    result = pd.concat(out_frames, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def build_cme_15m_cache(
    cache_dir: Path,
    output_path: Path,
    ticker_to_pair: Optional[dict] = None,
) -> int:
    """Aggregate all MBP-1 parquets in ``cache_dir`` into one 15-min returns parquet.

    Reads every ``mbp1_fx_*.parquet`` file under ``cache_dir``, concatenates them,
    runs :func:`aggregate_cme_to_15m_returns`, and writes the result to
    ``output_path``.

    Parameters
    ----------
    cache_dir : Path
        Directory containing ``mbp1_fx_*.parquet`` files.
    output_path : Path
        Destination parquet path (parent directory must exist).
    ticker_to_pair : dict, optional
        Defaults to ``DEFAULT_CME_TICKER_TO_PAIR``.

    Returns
    -------
    int
        Number of rows in the written cache. ``0`` if no files were found.
    """
    if ticker_to_pair is None:
        ticker_to_pair = dict(DEFAULT_CME_TICKER_TO_PAIR)

    cache_dir = Path(cache_dir)
    output_path = Path(output_path)

    files = sorted(cache_dir.glob("mbp1_fx_*.parquet"))
    if not files:
        return 0

    frames = []
    for fp in files:
        try:
            if fp.stat().st_size == 0:
                continue
        except OSError:
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        frames.append(df)

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True)
    result = aggregate_cme_to_15m_returns(combined, ticker_to_pair)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return len(result)


class FixAlphaModel:
    """Model the W-shaped intraday FX fix pattern for alpha generation.

    Parameters
    ----------
    fix_times_utc : dict, optional
        Mapping of fix name -> datetime.time in UTC.
        Defaults to Tokyo (00:55), ECB (13:15), London WMR (16:00).
    """

    def __init__(self, fix_times_utc: Optional[Dict[str, time]] = None):
        self.fix_times = fix_times_utc or dict(DEFAULT_FIX_TIMES)

    # ------------------------------------------------------------------
    # W-pattern computation from historical data
    # ------------------------------------------------------------------

    def compute_w_pattern(self, intraday_returns_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the average return per 15-min time-of-day bucket.

        This reveals the "W" shape: dips (USD appreciation = negative returns for
        EUR/USD etc.) in the 60 minutes before each fix, bounces after.

        Example output (EUR/USD):
            time_bucket | mean_return_bps | std_bps | t_stat
            00:00       | -0.02          | 5.1     | -0.06
            00:15       | -0.15          | 4.8     | -0.50
            00:30       | -0.45          | 5.0     | -1.43  <-- pre-Tokyo drift
            00:45       | -0.82          | 5.2     | -2.51  <-- strong pre-Tokyo
            01:00       |  0.35          | 5.5     |  1.01  <-- post-Tokyo reversal
            ...
            15:30       | -1.20          | 6.1     | -3.12  <-- pre-WMR (strongest)
            15:45       | -1.55          | 6.3     | -3.91  <-- peak pre-fix drift
            16:00       |  0.80          | 7.0     |  1.81  <-- post-WMR reversal

        Parameters
        ----------
        intraday_returns_df : pd.DataFrame
            Must have columns: [timestamp, pair, return_15m].
            timestamp should be timezone-aware or parseable as UTC.

        Returns
        -------
        pd.DataFrame
            Columns: [time_bucket, pair, mean_return, std_return, n_obs, t_stat].
        """
        df = intraday_returns_df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract 15-min bucket as HH:MM string
        df["time_bucket"] = df["timestamp"].dt.strftime("%H:%M")

        results = []
        for (bucket, pair), grp in df.groupby(["time_bucket", "pair"]):
            rets = grp["return_15m"].values
            n = len(rets)
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1) if n > 1 else np.nan
            t_stat = (mean_r / (std_r / np.sqrt(n))) if (std_r > 0 and n > 1) else 0.0
            results.append({
                "time_bucket": bucket,
                "pair": pair,
                "mean_return": mean_r,
                "std_return": std_r,
                "n_obs": n,
                "t_stat": t_stat,
            })

        return pd.DataFrame(results).sort_values(["pair", "time_bucket"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Time to next fix
    # ------------------------------------------------------------------

    def minutes_to_next_fix(self, current_utc_time: datetime) -> Tuple[float, str]:
        """Return (minutes_until_next_fix, fix_name).

        Example:
            current_utc_time = 15:30 UTC -> (30.0, 'london')
            current_utc_time = 00:40 UTC -> (15.0, 'tokyo')
            current_utc_time = 16:05 UTC -> next day Tokyo = (530.0, 'tokyo')

        Parameters
        ----------
        current_utc_time : datetime
            The current time in UTC.

        Returns
        -------
        tuple of (float, str)
            Minutes to next fix and the fix name.
        """
        current_t = current_utc_time.time() if isinstance(current_utc_time, datetime) else current_utc_time

        best_minutes = float("inf")
        best_fix = ""

        for fix_name, fix_time in self.fix_times.items():
            # Minutes from current to fix (same day)
            delta_min = (
                (fix_time.hour * 60 + fix_time.minute)
                - (current_t.hour * 60 + current_t.minute + current_t.second / 60.0)
            )
            if delta_min < 0:
                delta_min += 24 * 60  # wrap to next day

            if delta_min < best_minutes:
                best_minutes = delta_min
                best_fix = fix_name

        return best_minutes, best_fix

    # ------------------------------------------------------------------
    # Fix schedule for quoting
    # ------------------------------------------------------------------

    def fix_schedule(self, current_utc_time: datetime) -> Dict:
        """Generate quoting parameters based on proximity to the next fix.

        Returns a dict controlling spread, skew, and risk flags.

        Example at 15:45 UTC (15 min before London WMR fix):
            {
                'spread_multiplier': 0.8,       # tighten to capture flow
                'skew_direction': +1,            # accumulate USD (tighten bid for EUR/USD)
                'fix_proximity': 'pre_fix',
                'fix_name': 'london',
                'minutes_to_fix': 15.0,
                'adverse_selection_risk': 0.3,
            }

        Example at 16:01 UTC (1 min after London fix):
            {
                'spread_multiplier': 1.8,        # widen -- adverse selection peak
                'skew_direction': 0,              # neutral
                'fix_proximity': 'at_fix',
                'fix_name': 'london',
                'minutes_to_fix': 1439.0,
                'adverse_selection_risk': 0.9,
            }

        Parameters
        ----------
        current_utc_time : datetime
            Current UTC time.

        Returns
        -------
        dict
            Quoting schedule parameters.
        """
        minutes_to_fix, fix_name = self.minutes_to_next_fix(current_utc_time)

        # Also check minutes *since* most recent fix (for post-fix window)
        minutes_since = self._minutes_since_last_fix(current_utc_time)

        # Determine fix proximity zone
        if minutes_since is not None and minutes_since <= 5:
            # Within 5 min AFTER a fix -> "at_fix" (peak adverse selection)
            proximity = "at_fix"
            spread_mult = 1.5 + 0.1 * (5 - minutes_since)  # 1.5 to 2.0
            skew = 0  # no directional view right at fix
            adverse = 0.7 + 0.06 * (5 - minutes_since)  # 0.7 to 1.0

        elif minutes_since is not None and minutes_since <= 30:
            # 5-30 min after fix -> "post_fix" (reversal expected)
            proximity = "post_fix"
            spread_mult = 1.0 + 0.02 * (30 - minutes_since)  # 1.0 to 1.5
            skew = -1  # sell USD (reversal trades)
            adverse = 0.3 + 0.013 * (30 - minutes_since)  # ~0.3-0.7

        elif minutes_to_fix <= 60:
            # Within 60 min BEFORE next fix -> "pre_fix" (the alpha window)
            proximity = "pre_fix"
            # Tighter spread to capture flow, strongest near the fix
            spread_mult = max(0.7, 1.0 - 0.005 * (60 - minutes_to_fix))  # 0.7 to 1.0
            skew = +1  # accumulate USD (the Krohn et al. signal)
            adverse = 0.1 + 0.005 * (60 - minutes_to_fix)  # 0.1 to 0.4

        else:
            proximity = "neutral"
            spread_mult = 1.0
            skew = 0
            adverse = 0.1

        return {
            "spread_multiplier": round(spread_mult, 3),
            "skew_direction": skew,
            "fix_proximity": proximity,
            "fix_name": fix_name,
            "minutes_to_fix": round(minutes_to_fix, 1),
            "adverse_selection_risk": round(adverse, 3),
        }

    # ------------------------------------------------------------------
    # Fix-window returns analysis
    # ------------------------------------------------------------------

    def compute_fix_returns(self, price_history_df: pd.DataFrame) -> pd.DataFrame:
        """Compute average returns in windows around each fix.

        Windows (minutes relative to fix): [-60,-30], [-30,-15], [-15,0],
        [0,+15], [+15,+30], [+30,+60].

        This is THE KEY EMPIRICAL RESULT: pre-fix returns should be negative
        for EUR/USD, GBP/USD, AUD/USD (USD appreciation) and positive for
        JPY/USD (if quoted as JPY per USD).

        Example output:
            fix_name | pair    | window     | mean_return_bps | t_stat
            london   | EUR/USD | [-60,-30]  | -0.45          | -1.82
            london   | EUR/USD | [-30,-15]  | -0.82          | -2.91
            london   | EUR/USD | [-15,0]    | -1.55          | -3.91
            london   | EUR/USD | [0,+15]    |  0.80          |  1.81
            london   | EUR/USD | [+15,+30]  |  0.45          |  1.22
            london   | EUR/USD | [+30,+60]  |  0.20          |  0.55

        Parameters
        ----------
        price_history_df : pd.DataFrame
            Columns: [timestamp, pair, mid_price]. High-frequency intraday data.
            Timestamps should be UTC. Need at least 1-min resolution for best results.

        Returns
        -------
        pd.DataFrame
            Average returns per window, per fix, per pair.
        """
        df = price_history_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values(["pair", "timestamp"]).reset_index(drop=True)

        windows = [
            (-60, -30, "[-60,-30]"),
            (-30, -15, "[-30,-15]"),
            (-15, 0, "[-15,0]"),
            (0, 15, "[0,+15]"),
            (15, 30, "[+15,+30]"),
            (30, 60, "[+30,+60]"),
        ]

        results = []
        for pair in df["pair"].unique():
            pair_df = df[df["pair"] == pair].set_index("timestamp")

            for fix_name, fix_time in self.fix_times.items():
                # Get unique dates
                dates = pair_df.index.date
                unique_dates = sorted(set(dates))

                for window_start, window_end, window_label in windows:
                    window_returns = []

                    for d in unique_dates:
                        fix_dt = datetime.combine(d, fix_time)
                        t_start = fix_dt + timedelta(minutes=window_start)
                        t_end = fix_dt + timedelta(minutes=window_end)

                        mask = (pair_df.index >= pd.Timestamp(t_start)) & (
                            pair_df.index <= pd.Timestamp(t_end)
                        )
                        window_data = pair_df[mask]

                        if len(window_data) >= 2:
                            p_start = window_data["mid_price"].iloc[0]
                            p_end = window_data["mid_price"].iloc[-1]
                            if p_start != 0:
                                ret_bps = (p_end / p_start - 1.0) * 10_000
                                window_returns.append(ret_bps)

                    if len(window_returns) > 0:
                        arr = np.array(window_returns)
                        mean_r = np.mean(arr)
                        std_r = np.std(arr, ddof=1) if len(arr) > 1 else np.nan
                        t_stat = (
                            mean_r / (std_r / np.sqrt(len(arr)))
                            if (std_r and std_r > 0 and len(arr) > 1)
                            else 0.0
                        )
                        results.append({
                            "fix_name": fix_name,
                            "pair": pair,
                            "window": window_label,
                            "mean_return_bps": round(mean_r, 4),
                            "std_bps": round(std_r, 4) if not np.isnan(std_r) else np.nan,
                            "n_days": len(arr),
                            "t_stat": round(t_stat, 3),
                        })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Real data loader (yfinance) with synthetic fallback
    # ------------------------------------------------------------------

    def load_real_intraday_data(
        self,
        pairs: Optional[List[str]] = None,
        interval: str = "1h",
        period: str = "60d",
    ) -> Optional[pd.DataFrame]:
        """Load REAL intraday FX data from Yahoo Finance.

        Returns data in the same format as generate_realistic_intraday_data():
        columns [timestamp, pair, return_15m].

        For 1h bars (recommended, 2 years of history), the hourly returns are
        split into 4 x 15-min sub-periods to match the fix analysis framework.

        Example
        -------
        >>> model = FixAlphaModel()
        >>> df = model.load_real_intraday_data(["EUR/USD"], interval="1h", period="2y")
        >>> len(df)  # ~48,000 rows (2 years * 252 days * 96 buckets)
        """
        try:
            from shared.yfinance_loader import fetch_fx_intraday, convert_to_15min_returns
        except ImportError:
            return None

        if pairs is None:
            pairs = list(PAIR_DAILY_VOL_PIPS.keys())

        raw = fetch_fx_intraday(pairs=pairs, interval=interval, period=period)
        if raw is None or raw.empty:
            return None

        if interval == "1h":
            # Convert 1h bars to 15-min returns
            result = convert_to_15min_returns(raw)
        elif interval in ("15m", "5m"):
            result = raw[["timestamp", "pair", f"return_{interval}"]].copy()
            result = result.rename(columns={f"return_{interval}": "return_15m"})
        else:
            result = raw[["timestamp", "pair"]].copy()
            ret_col = [c for c in raw.columns if c.startswith("return_")]
            if ret_col:
                result["return_15m"] = raw[ret_col[0]]
            else:
                return None

        return result.dropna(subset=["return_15m"]).reset_index(drop=True)

    def get_intraday_data(
        self,
        pairs: Optional[List[str]] = None,
        n_days: int = 252,
        seed: int = 42,
        prefer_real: bool = True,
        prefer_cme: bool = True,
    ) -> Tuple[pd.DataFrame, str]:
        """Get intraday data: tries CME aggregated data, then Yahoo Finance, then synthetic.

        Returns
        -------
        (DataFrame, source_label) where source_label is one of
        ``"cme_aggregated"``, ``"yahoo_finance"``, ``"yahoo_finance_15m"``,
        or ``"synthetic"``.

        Example
        -------
        >>> model = FixAlphaModel()
        >>> df, source = model.get_intraday_data(prefer_real=True)
        >>> print(f"Got {len(df)} rows from {source}")
        Got 96768 rows from yahoo_finance
        """
        if prefer_cme:
            cme_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "cme_15m_cache.parquet"
            )
            if cme_path.exists():
                try:
                    cme_df = pd.read_parquet(cme_path)
                    if pairs is not None:
                        cme_df = cme_df[cme_df["pair"].isin(pairs)]
                    cme_df = cme_df.reset_index(drop=True)
                    if not cme_df.empty:
                        return cme_df, "cme_aggregated"
                except Exception:
                    pass

        if prefer_real:
            real = self.load_real_intraday_data(pairs=pairs, interval="1h", period="2y")
            if real is not None and len(real) > 1000:
                return real, "yahoo_finance"

            # Try 15m for recent 60 days
            real_15m = self.load_real_intraday_data(pairs=pairs, interval="15m", period="60d")
            if real_15m is not None and len(real_15m) > 500:
                return real_15m, "yahoo_finance_15m"

        synthetic = self.generate_realistic_intraday_data(pairs=pairs, n_days=n_days, seed=seed)
        return synthetic, "synthetic"

    # ------------------------------------------------------------------
    # Synthetic intraday data generator (fallback)
    # ------------------------------------------------------------------

    def generate_realistic_intraday_data(
        self,
        pairs: Optional[List[str]] = None,
        n_days: int = 252,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate realistic 15-min intraday FX returns embedding the W-shape pattern.

        Calibrated parameters from Krohn et al. (2024):
            - Pre-fix drift: ~1-3 bps per fix window (annualized Sharpe ~2-5)
            - Noise: ~5-8 bps per 15-min bucket for majors
            - Post-fix reversal: ~0.5-2 bps
            - Time-of-day vol: low Asia, high London/NY overlap

        Example output rows:
            timestamp           | pair    | return_15m
            2024-01-02 00:00:00 | EUR/USD | -0.00023
            2024-01-02 00:15:00 | EUR/USD | 0.00015
            2024-01-02 00:30:00 | EUR/USD | -0.00045  (pre-Tokyo drift starting)
            ...

        Parameters
        ----------
        pairs : list of str, optional
            Currency pairs. Defaults to G4 majors.
        n_days : int
            Number of trading days to simulate.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: [timestamp, pair, return_15m] with 96 rows per day per pair.
        """
        rng = np.random.default_rng(seed)

        if pairs is None:
            pairs = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]

        vol_pattern = _build_intraday_vol_pattern()  # 96 elements

        # Pre-fix drift magnitudes (in return space, e.g. -0.0002 = -2 bps)
        # Strongest for London WMR, weakest for Tokyo
        fix_drift_magnitudes = {
            "tokyo": -0.00010,   # -1.0 bps pre-Tokyo
            "ecb": -0.00015,     # -1.5 bps pre-ECB
            "london": -0.00025,  # -2.5 bps pre-London (strongest, per Krohn et al.)
        }

        # Post-fix reversal (about 50-70% of drift)
        reversal_fractions = {"tokyo": 0.50, "ecb": 0.60, "london": 0.65}

        # Map fix times to 15-min bucket indices
        fix_buckets = {}
        for name, ft in self.fix_times.items():
            bucket_idx = (ft.hour * 60 + ft.minute) // 15
            fix_buckets[name] = bucket_idx  # e.g. london -> 64

        # Build the deterministic drift pattern (96 buckets)
        drift_pattern = np.zeros(96)
        for fix_name, fix_bucket in fix_buckets.items():
            mag = fix_drift_magnitudes.get(fix_name, -0.00015)
            reversal = reversal_fractions.get(fix_name, 0.60)

            # Pre-fix drift: ramp up over 4 buckets (60 min) before fix
            for offset in range(1, 5):  # buckets -4, -3, -2, -1 before fix
                idx = (fix_bucket - offset) % 96
                # Stronger closer to fix: weights [0.15, 0.25, 0.30, 0.30]
                weight = [0.30, 0.30, 0.25, 0.15][offset - 1]
                drift_pattern[idx] += mag * weight

            # Post-fix reversal over 2-3 buckets
            for offset in range(0, 3):
                idx = (fix_bucket + offset) % 96
                rev_weight = [0.45, 0.35, 0.20][offset]
                drift_pattern[idx] += -mag * reversal * rev_weight

        # Generate dates (weekdays only)
        base_date = datetime(2024, 1, 2)  # First business day of 2024
        dates = []
        d = base_date
        while len(dates) < n_days:
            if d.weekday() < 5:  # Mon-Fri
                dates.append(d)
            d += timedelta(days=1)

        rows = []
        for pair in pairs:
            daily_vol = PAIR_DAILY_VOL_PIPS.get(pair, 60)
            # Convert daily vol in pips to return: 50 pips on EUR/USD at 1.0850 ~ 0.0046
            pip_size = 0.01 if "JPY" in pair else 0.0001
            daily_vol_return = daily_vol * pip_size  # e.g. 50 * 0.0001 = 0.005

            # Per-bucket vol = daily_vol / sqrt(96) * vol_pattern_weight
            base_bucket_vol = daily_vol_return / np.sqrt(96)

            # JPY/USD is quoted inversely, so USD appreciation = positive return
            pair_sign = 1.0 if "JPY" in pair else 1.0  # keep sign the same for simplicity; drift is negative for EUR/USD

            for day in dates:
                for bucket in range(96):
                    hour = bucket // 4
                    minute = (bucket % 4) * 15
                    ts = day.replace(hour=hour, minute=minute, second=0)

                    # Noise: normal with time-of-day vol scaling
                    noise = rng.normal(0, base_bucket_vol * np.sqrt(vol_pattern[bucket]))

                    # Signal: fix-driven drift
                    signal = drift_pattern[bucket]

                    # Scale signal slightly by pair (GBP is noisier, AUD more)
                    pair_scale = {"EUR/USD": 1.0, "GBP/USD": 0.9, "JPY/USD": 0.8, "AUD/USD": 1.1}
                    signal *= pair_scale.get(pair, 1.0)

                    ret = signal + noise

                    rows.append({
                        "timestamp": ts,
                        "pair": pair,
                        "return_15m": ret,
                    })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _minutes_since_last_fix(self, current_utc_time: datetime) -> Optional[float]:
        """Minutes since the most recent fix, or None if > 60 min ago."""
        current_t = current_utc_time.time() if isinstance(current_utc_time, datetime) else current_utc_time
        current_min = current_t.hour * 60 + current_t.minute + current_t.second / 60.0

        best = None
        for fix_name, fix_time in self.fix_times.items():
            fix_min = fix_time.hour * 60 + fix_time.minute
            elapsed = current_min - fix_min
            if elapsed < 0:
                elapsed += 24 * 60  # wrap around

            if elapsed <= 60:
                if best is None or elapsed < best:
                    best = elapsed

        return best


# ======================================================================
# Composite Alpha Model
# ======================================================================

class CompositeAlphaModel:
    """Combine fix-timing alpha with carry, momentum, and mean-reversion signals.

    The composite signal drives quote skewing in pips:
        composite_skew = fix_wt * fix_signal
                       + carry_wt * carry_signal
                       + momentum_wt * momentum_signal
                       + mr_wt * mean_reversion_signal

    All signals are z-scored to [-1, +1] range, then the weighted sum is
    scaled by max_skew_pips and clipped.

    Parameters
    ----------
    fix_wt : float
        Weight on fix-timing signal (default 0.50 -- it's THE alpha).
    carry_wt : float
        Weight on interest rate differential (default 0.25).
    momentum_wt : float
        Weight on FX momentum (default 0.15).
    mr_wt : float
        Weight on mean-reversion signal (default 0.10).
    max_skew_pips : float
        Maximum skew in pips applied to quotes (default 3.0).

    Example:
        fix_signal = +0.8 (strong pre-fix, accumulate USD)
        carry_signal = +0.3 (slightly positive carry for high-yielder)
        momentum_signal = -0.2 (slight bearish momentum)
        mr_signal = +0.1 (near fair value)
        composite = 0.50*0.8 + 0.25*0.3 + 0.15*(-0.2) + 0.10*0.1 = 0.455
        skew = 0.455 * 3.0 = 1.365 pips (tighten bid, widen offer for EUR/USD)
    """

    def __init__(
        self,
        fix_wt: float = 0.50,
        carry_wt: float = 0.25,
        momentum_wt: float = 0.15,
        mr_wt: float = 0.10,
        max_skew_pips: float = 3.0,
    ):
        self.fix_wt = fix_wt
        self.carry_wt = carry_wt
        self.momentum_wt = momentum_wt
        self.mr_wt = mr_wt
        self.max_skew_pips = max_skew_pips
        self.fix_alpha = FixAlphaModel()

    def compute_signals(
        self,
        rates_history: Optional[pd.DataFrame] = None,
        intraday_data: Optional[pd.DataFrame] = None,
        current_utc_time: Optional[datetime] = None,
    ) -> Dict:
        """Compute all four alpha signals.

        Parameters
        ----------
        rates_history : pd.DataFrame, optional
            Daily FX rate history with columns [date, pair, close, rate_domestic, rate_foreign].
            Used for carry, momentum, and mean-reversion signals.
        intraday_data : pd.DataFrame, optional
            Intraday 15-min return data for fix signal calibration.
        current_utc_time : datetime, optional
            Current UTC time. Defaults to now.

        Returns
        -------
        dict
            Keys: fix_signal, carry_signal, momentum_signal, mean_reversion_signal,
            composite_skew (in pips), per pair.
        """
        if current_utc_time is None:
            current_utc_time = datetime.utcnow()

        # 1. Fix signal (time-of-day)
        fix_sched = self.fix_alpha.fix_schedule(current_utc_time)
        fix_signal = fix_sched["skew_direction"]  # +1, -1, or 0

        # Scale by proximity: stronger when closer to fix
        if fix_sched["fix_proximity"] == "pre_fix":
            minutes = fix_sched["minutes_to_fix"]
            fix_signal *= max(0.3, 1.0 - minutes / 60.0)  # 0.3 at 60min, 1.0 at 0min

        # 2. Carry signal: rate differential z-scored
        carry_signal = self._carry_signal(rates_history)

        # 3. Momentum signal: 1W and 1M returns
        momentum_signal = self._momentum_signal(rates_history)

        # 4. Mean-reversion signal: deviation from 200-day MA
        mr_signal = self._mean_reversion_signal(rates_history)

        # Composite
        composite_raw = (
            self.fix_wt * fix_signal
            + self.carry_wt * carry_signal
            + self.momentum_wt * momentum_signal
            + self.mr_wt * mr_signal
        )
        composite_skew = np.clip(composite_raw * self.max_skew_pips,
                                 -self.max_skew_pips, self.max_skew_pips)

        return {
            "fix_signal": round(fix_signal, 4),
            "carry_signal": round(carry_signal, 4),
            "momentum_signal": round(momentum_signal, 4),
            "mean_reversion_signal": round(mr_signal, 4),
            "composite_raw": round(composite_raw, 4),
            "composite_skew_pips": round(composite_skew, 4),
            "fix_schedule": fix_sched,
        }

    def skew_for_rfq(
        self,
        rates_history: Optional[pd.DataFrame],
        pair: str,
        direction: str,
        current_utc_time: Optional[datetime] = None,
    ) -> float:
        """Return the skew in pips to apply to an RFQ quote.

        Positive skew = tighten bid (encourage buying base currency = selling USD).
        Negative skew = tighten offer (encourage selling base currency = buying USD).

        Example:
            Pre-London fix, EUR/USD, client wants to buy EUR:
            composite_skew = +1.5 pips (we want to accumulate USD, so tighten our offer
            to make it attractive for the client to sell EUR to us)
            -> For a "buy_base" RFQ, we SUBTRACT skew from offer (tighten)
            -> Return: -1.5 pips (client gets a better offer, we accumulate USD)

        Parameters
        ----------
        rates_history : pd.DataFrame or None
            Historical rates data.
        pair : str
            Currency pair, e.g. 'EUR/USD'.
        direction : str
            Client's direction: 'buy_base' or 'sell_base'.
        current_utc_time : datetime, optional
            Current UTC time.

        Returns
        -------
        float
            Skew in pips (positive = tighten bid, negative = tighten offer).
        """
        signals = self.compute_signals(rates_history, current_utc_time=current_utc_time)
        skew = signals["composite_skew_pips"]

        # Adjust sign based on client direction
        # If skew > 0 (we want to buy USD = sell base), and client wants to buy_base,
        # we TIGHTEN our offer (make it cheaper for client) -> negative adjustment
        if direction == "buy_base":
            return -skew
        else:
            return skew

    # ------------------------------------------------------------------
    # Individual signal computations
    # ------------------------------------------------------------------

    def _carry_signal(self, rates_history: Optional[pd.DataFrame]) -> float:
        """Interest rate differential signal, z-scored.

        carry = (rate_foreign - rate_domestic)
        Positive carry means foreign currency yields more -> buy foreign (sell USD).
        Z-score using trailing 252-day history of carry.

        Example:
            AUD rate = 4.25%, USD rate = 5.25% -> carry = -1.00%
            If historical carry mean = -0.50%, std = 0.80%
            z-score = (-1.00 - (-0.50)) / 0.80 = -0.625
            Clipped to [-1, 1] -> -0.625 (lean towards selling AUD)
        """
        if rates_history is None or len(rates_history) < 10:
            return 0.0

        try:
            if "rate_domestic" in rates_history.columns and "rate_foreign" in rates_history.columns:
                carry = rates_history["rate_foreign"] - rates_history["rate_domestic"]
                current_carry = carry.iloc[-1]
                carry_mean = carry.mean()
                carry_std = carry.std()
                if carry_std > 0:
                    z = (current_carry - carry_mean) / carry_std
                    return float(np.clip(z, -1.0, 1.0))
        except Exception:
            pass

        return 0.0

    def _momentum_signal(self, rates_history: Optional[pd.DataFrame]) -> float:
        """FX momentum signal: blend of 1-week and 1-month returns, z-scored.

        Example:
            EUR/USD 1W return = +0.3%, 1M return = +1.2%
            Blended momentum = 0.5 * 0.3% + 0.5 * 1.2% = +0.75%
            If historical blended mean = 0.0%, std = 0.5%
            z-score = 0.75 / 0.5 = 1.5 -> clipped to 1.0
        """
        if rates_history is None or len(rates_history) < 25:
            return 0.0

        try:
            if "close" in rates_history.columns:
                prices = rates_history["close"].values
                # 1-week (~5 days) return
                ret_1w = (prices[-1] / prices[-5] - 1.0) if len(prices) >= 5 else 0.0
                # 1-month (~22 days) return
                ret_1m = (prices[-1] / prices[-22] - 1.0) if len(prices) >= 22 else 0.0

                blended = 0.5 * ret_1w + 0.5 * ret_1m

                # Z-score using rolling returns
                if len(prices) >= 25:
                    rolling_rets = pd.Series(prices).pct_change(5).dropna()
                    if rolling_rets.std() > 0:
                        z = blended / rolling_rets.std()
                        return float(np.clip(z, -1.0, 1.0))
        except Exception:
            pass

        return 0.0

    def _mean_reversion_signal(self, rates_history: Optional[pd.DataFrame]) -> float:
        """Mean-reversion signal: deviation from 200-day MA, z-scored.

        Example:
            EUR/USD current = 1.0850, 200-day MA = 1.0920
            Deviation = (1.0850 - 1.0920) / 1.0920 = -0.64%
            If std of deviations = 1.5% -> z = -0.64 / 1.5 = -0.43
            Signal = +0.43 (expect reversion UP, so buy EUR = sell USD)
            Note: sign is flipped because MR signal opposes the deviation.
        """
        if rates_history is None or len(rates_history) < 200:
            return 0.0

        try:
            if "close" in rates_history.columns:
                prices = rates_history["close"].values
                ma_200 = np.mean(prices[-200:])
                current = prices[-1]

                if ma_200 > 0:
                    deviation = (current - ma_200) / ma_200
                    # Historical deviations for z-scoring
                    devs = []
                    for i in range(200, len(prices)):
                        ma_i = np.mean(prices[i - 200:i])
                        if ma_i > 0:
                            devs.append((prices[i] - ma_i) / ma_i)
                    if len(devs) > 1:
                        std_dev = np.std(devs)
                        if std_dev > 0:
                            z = deviation / std_dev
                            # Flip sign: positive deviation -> expect reversion down -> negative signal
                            return float(np.clip(-z, -1.0, 1.0))
        except Exception:
            pass

        return 0.0
