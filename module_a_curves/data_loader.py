"""
FX Data Loader -- Three-tier fallback: Live FRED API -> Parquet cache -> Embedded snapshot.

Fetches FX spot rates, interest rate data, and historical price series for the
FX fix-aware market-making system.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED Series Mappings
# ---------------------------------------------------------------------------
FRED_FX_SERIES: Dict[str, str] = {
    "EUR/USD": "DEXUSEU",
    "GBP/USD": "DEXUSUK",
    "JPY/USD": "DEXJPUS",
    "AUD/USD": "DEXUSAL",
}

FRED_RATE_SERIES: Dict[str, str] = {
    "SOFR": "SOFR",
    "SOFR30DAYAVG": "SOFR30DAYAVG",
    "SOFR90DAYAVG": "SOFR90DAYAVG",
    "DGS2": "DGS2",
    "DGS5": "DGS5",
    "DGS10": "DGS10",
    "DGS30": "DGS30",
    "ECBESTRVOLWGTTRMDMNRT": "ECBESTRVOLWGTTRMDMNRT",
    "IUDSOIA": "IUDSOIA",
}

# ---------------------------------------------------------------------------
# Embedded Fallback Snapshot (2026-03-05)
# ---------------------------------------------------------------------------
FALLBACK_FX_SPOT: Dict[str, float] = {
    "EUR/USD": 1.0832,
    "GBP/USD": 1.2714,
    "JPY/USD": 0.006667,
    "AUD/USD": 0.6358,
    "CAD/USD": 0.6993,
    "CHF/USD": 1.1186,
    "NZD/USD": 0.5712,
}

FALLBACK_RATES: Dict[str, float] = {
    "SOFR": 0.0433,
    "SOFR30DAYAVG": 0.0432,
    "SOFR90DAYAVG": 0.0431,
    "DGS2": 0.0412,
    "DGS5": 0.0428,
    "DGS10": 0.0455,
    "DGS30": 0.0472,
    "ECBESTRVOLWGTTRMDMNRT": 0.0290,
    "IUDSOIA": 0.0450,
}

# Calibrated annualised volatilities for GBM fallback history generation
_GBM_VOLS: Dict[str, float] = {
    "EUR/USD": 0.07,
    "GBP/USD": 0.08,
    "JPY/USD": 0.09,
    "AUD/USD": 0.11,
}

# ---------------------------------------------------------------------------
# Central Bank Meeting Dates (deterministic public calendar, 2026)
# ---------------------------------------------------------------------------
_CB_DATES_2026: Dict[str, List[str]] = {
    "FOMC": [
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ],
    "ECB": [
        "2026-01-22", "2026-03-05", "2026-04-16", "2026-06-04",
        "2026-07-16", "2026-09-10", "2026-10-29", "2026-12-17",
    ],
    "BOE": [
        "2026-02-05", "2026-03-19", "2026-05-07", "2026-06-18",
        "2026-08-06", "2026-09-17", "2026-11-05", "2026-12-17",
    ],
}


class FXDataLoader:
    """Three-tier FX data loader: FRED API -> Parquet cache -> embedded snapshot.

    Example
    -------
    >>> loader = FXDataLoader(fred_api_key="abc123")
    >>> spots = loader.get_fx_spot_rates("2026-03-05")
    >>> spots["EUR/USD"]
    1.0832
    """

    def __init__(self, fred_api_key: Optional[str] = None) -> None:
        """Initialise the data loader.

        Parameters
        ----------
        fred_api_key : str or None
            FRED API key.  If *None*, falls back to ``config.get_fred_api_key()``
            and then to the embedded snapshot.
        """
        if fred_api_key is None:
            try:
                from config import get_fred_api_key
                fred_api_key = get_fred_api_key()
            except ImportError:
                pass
        self.fred_api_key: Optional[str] = fred_api_key

        # Cache directory: <project_root>/data/cache/
        self._project_root = Path(__file__).resolve().parent.parent
        self._cache_dir = self._project_root / "data" / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._fx_cache_path = self._cache_dir / "fx_spot_cache.parquet"
        self._rates_cache_path = self._cache_dir / "rates_cache.parquet"
        self._history_cache_path = self._cache_dir / "fx_history_cache.parquet"

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def get_fx_spot_rates(self, date_str: str) -> Dict[str, float]:
        """Return FX spot rates for *date_str* (YYYY-MM-DD).

        Returns a dict like ``{"EUR/USD": 1.0832, "GBP/USD": 1.2714, ...}``.

        Tier 1: FRED API live pull for each series on the requested date.
        Tier 2: Parquet cache lookup.
        Tier 3: Embedded fallback snapshot (2026-03-05 levels).
        """
        # Tier 1 -- live FRED
        if self.fred_api_key:
            try:
                result = self._fetch_fred_fx(date_str)
                if result:
                    self._save_fx_cache(date_str, result)
                    logger.info("FX spots from FRED API for %s", date_str)
                    return result
            except Exception as exc:
                logger.warning("FRED API FX fetch failed: %s", exc)

        # Tier 2 -- parquet cache
        cached = self._load_fx_cache(date_str)
        if cached:
            logger.info("FX spots from parquet cache for %s", date_str)
            return cached

        # Tier 2.5 -- yfinance latest close (real data, no API key)
        try:
            from shared.yfinance_loader import fetch_fx_daily
            yf_df = fetch_fx_daily(pairs=list(FRED_FX_SERIES.keys()), period="5d")
            if yf_df is not None and not yf_df.empty:
                latest = yf_df.iloc[-1]
                result = {pair: float(latest[pair]) for pair in yf_df.columns if pair in FRED_FX_SERIES}
                if result:
                    logger.info("FX spots from Yahoo Finance (latest close)")
                    return result
        except Exception as exc:
            logger.warning("yfinance spot fetch failed: %s", exc)

        # Tier 3 -- embedded fallback
        logger.info("FX spots from embedded fallback snapshot (2026-03-05)")
        return dict(FALLBACK_FX_SPOT)

    def get_ois_rates(self, date_str: str) -> Dict[str, float]:
        """Return OIS / policy / treasury rates for curve construction.

        Returns a dict like ``{"SOFR": 0.0433, "DGS10": 0.0455, ...}``.
        Same three-tier fallback as FX spots.
        """
        # Tier 1 -- live FRED
        if self.fred_api_key:
            try:
                result = self._fetch_fred_rates(date_str)
                if result:
                    self._save_rates_cache(date_str, result)
                    logger.info("OIS rates from FRED API for %s", date_str)
                    return result
            except Exception as exc:
                logger.warning("FRED API rates fetch failed: %s", exc)

        # Tier 2 -- parquet cache
        cached = self._load_rates_cache(date_str)
        if cached:
            logger.info("OIS rates from parquet cache for %s", date_str)
            return cached

        # Tier 3 -- embedded fallback
        logger.info("OIS rates from embedded fallback snapshot (2026-03-05)")
        return dict(FALLBACK_RATES)

    def fetch_history(
        self,
        start: str,
        end: str,
        pairs: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of daily FX spot rates between *start* and *end*.

        Columns are pair names (e.g. ``EUR/USD``), index is ``DatetimeIndex``
        of business days.

        If FRED is unavailable, generates synthetic data via Geometric Brownian
        Motion calibrated to realistic annualised vols:
        EUR 7%, GBP 8%, JPY 9%, AUD 11%.
        """
        if pairs is None:
            pairs = list(FRED_FX_SERIES.keys())

        # Tier 1 -- live FRED
        if self.fred_api_key:
            try:
                df = self._fetch_fred_history(start, end, pairs)
                if df is not None and not df.empty:
                    df.to_parquet(self._history_cache_path)
                    logger.info("FX history from FRED API (%s -> %s)", start, end)
                    return df
            except Exception as exc:
                logger.warning("FRED history fetch failed: %s", exc)

        # Tier 2 -- parquet cache
        if self._history_cache_path.exists():
            try:
                df = pd.read_parquet(self._history_cache_path)
                mask = (df.index >= start) & (df.index <= end)
                subset = df.loc[mask, [p for p in pairs if p in df.columns]]
                if not subset.empty:
                    logger.info("FX history from parquet cache")
                    return subset
            except Exception as exc:
                logger.warning("Parquet history cache read failed: %s", exc)

        # Tier 2.5 -- yfinance daily (real market data, no API key needed)
        try:
            from shared.yfinance_loader import fetch_fx_daily
            yf_df = fetch_fx_daily(pairs=pairs, start=start, end=end)
            if yf_df is not None and not yf_df.empty and len(yf_df) > 10:
                yf_df.to_parquet(self._history_cache_path)
                logger.info("FX history from Yahoo Finance (%s -> %s)", start, end)
                return yf_df
        except Exception as exc:
            logger.warning("yfinance daily fetch failed: %s", exc)

        # Tier 3 -- synthetic GBM
        logger.info("FX history from synthetic GBM generator")
        return self._generate_gbm_history(start, end, pairs)

    def fetch_central_bank_dates(self, year: int = 2026) -> Dict[str, List[str]]:
        """Return deterministic central-bank meeting dates.

        Parameters
        ----------
        year : int
            Calendar year.  Only 2026 has embedded dates; other years
            return an empty dict.

        Returns
        -------
        dict
            ``{"FOMC": ["2026-01-28", ...], "ECB": [...], "BOE": [...]}``.
        """
        if year == 2026:
            return {k: list(v) for k, v in _CB_DATES_2026.items()}
        logger.warning("No embedded central-bank dates for year %d", year)
        return {"FOMC": [], "ECB": [], "BOE": []}

    # ------------------------------------------------------------------
    # FRED API Helpers
    # ------------------------------------------------------------------

    def _fred_get_value(self, series_id: str, date_str: str) -> Optional[float]:
        """Fetch the most recent observation for *series_id* on or before *date_str*."""
        import urllib.request
        import json

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={self.fred_api_key}"
            f"&file_type=json"
            f"&observation_start={date_str}"
            f"&observation_end={date_str}"
            f"&sort_order=desc"
            f"&limit=1"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        obs = data.get("observations", [])
        if obs and obs[0]["value"] != ".":
            return float(obs[0]["value"])
        return None

    def _fred_get_series(
        self, series_id: str, start: str, end: str
    ) -> Optional[pd.Series]:
        """Fetch a full time-series from FRED between *start* and *end*."""
        import urllib.request
        import json

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={self.fred_api_key}"
            f"&file_type=json"
            f"&observation_start={start}"
            f"&observation_end={end}"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        obs = data.get("observations", [])
        if not obs:
            return None
        dates, values = [], []
        for o in obs:
            if o["value"] != ".":
                dates.append(pd.Timestamp(o["date"]))
                values.append(float(o["value"]))
        if not dates:
            return None
        return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)

    def _fetch_fred_fx(self, date_str: str) -> Optional[Dict[str, float]]:
        """Fetch FX spot rates from FRED for a single date."""
        result: Dict[str, float] = {}
        for pair, series_id in FRED_FX_SERIES.items():
            val = self._fred_get_value(series_id, date_str)
            if val is not None:
                # FRED DEXJPUS is JPY per USD; convert to USD per JPY
                if pair == "JPY/USD":
                    val = 1.0 / val  # e.g. 1/150.0 = 0.006667
                result[pair] = round(val, 6)
        return result if result else None

    def _fetch_fred_rates(self, date_str: str) -> Optional[Dict[str, float]]:
        """Fetch OIS / treasury rates from FRED for a single date."""
        result: Dict[str, float] = {}
        for name, series_id in FRED_RATE_SERIES.items():
            val = self._fred_get_value(series_id, date_str)
            if val is not None:
                # FRED reports rates in percent; convert to decimal
                result[name] = round(val / 100.0, 6)
        return result if result else None

    def _fetch_fred_history(
        self, start: str, end: str, pairs: List[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch historical FX spot series from FRED."""
        frames: Dict[str, pd.Series] = {}
        for pair in pairs:
            series_id = FRED_FX_SERIES.get(pair)
            if series_id is None:
                continue
            s = self._fred_get_series(series_id, start, end)
            if s is not None:
                if pair == "JPY/USD":
                    s = 1.0 / s
                frames[pair] = s
        if not frames:
            return None
        df = pd.DataFrame(frames)
        df.index.name = "date"
        return df.dropna(how="all")

    # ------------------------------------------------------------------
    # Parquet Cache Helpers
    # ------------------------------------------------------------------

    def _save_fx_cache(self, date_str: str, data: Dict[str, float]) -> None:
        """Append FX spot data to the parquet cache."""
        row = pd.DataFrame([data], index=pd.DatetimeIndex([date_str], name="date"))
        if self._fx_cache_path.exists():
            existing = pd.read_parquet(self._fx_cache_path)
            row = pd.concat([existing, row])
            row = row[~row.index.duplicated(keep="last")]
        row.to_parquet(self._fx_cache_path)

    def _load_fx_cache(self, date_str: str) -> Optional[Dict[str, float]]:
        """Load FX spot data from the parquet cache for a given date."""
        if not self._fx_cache_path.exists():
            return None
        try:
            df = pd.read_parquet(self._fx_cache_path)
            ts = pd.Timestamp(date_str)
            if ts in df.index:
                return df.loc[ts].dropna().to_dict()
        except Exception:
            pass
        return None

    def _save_rates_cache(self, date_str: str, data: Dict[str, float]) -> None:
        """Append rates data to the parquet cache."""
        row = pd.DataFrame([data], index=pd.DatetimeIndex([date_str], name="date"))
        if self._rates_cache_path.exists():
            existing = pd.read_parquet(self._rates_cache_path)
            row = pd.concat([existing, row])
            row = row[~row.index.duplicated(keep="last")]
        row.to_parquet(self._rates_cache_path)

    def _load_rates_cache(self, date_str: str) -> Optional[Dict[str, float]]:
        """Load rates data from the parquet cache for a given date."""
        if not self._rates_cache_path.exists():
            return None
        try:
            df = pd.read_parquet(self._rates_cache_path)
            ts = pd.Timestamp(date_str)
            if ts in df.index:
                return df.loc[ts].dropna().to_dict()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Synthetic GBM History Generator
    # ------------------------------------------------------------------

    def _generate_gbm_history(
        self, start: str, end: str, pairs: List[str]
    ) -> pd.DataFrame:
        """Generate synthetic daily FX prices via Geometric Brownian Motion.

        Uses calibrated annualised volatilities and zero drift (risk-neutral)
        starting from the embedded fallback spot rates.  Only business days
        are generated.

        For example, with EUR/USD starting at 1.0832 and 7% annual vol,
        daily vol = 0.07 / sqrt(252) ~ 0.00441, so a typical daily move is
        about 0.0048 (roughly 5 pips).
        """
        bdays = pd.bdate_range(start=start, end=end)
        if bdays.empty:
            return pd.DataFrame()

        rng = np.random.default_rng(seed=42)
        n_days = len(bdays)
        data: Dict[str, np.ndarray] = {}

        for pair in pairs:
            s0 = FALLBACK_FX_SPOT.get(pair)
            vol = _GBM_VOLS.get(pair)
            if s0 is None or vol is None:
                continue

            daily_vol = vol / np.sqrt(252.0)
            # GBM: S(t+1) = S(t) * exp(-0.5*sigma_d^2 + sigma_d * Z)
            z = rng.standard_normal(n_days)
            log_returns = -0.5 * daily_vol**2 + daily_vol * z
            log_prices = np.log(s0) + np.cumsum(log_returns)
            data[pair] = np.exp(log_prices)

        df = pd.DataFrame(data, index=bdays)
        df.index.name = "date"
        return df
