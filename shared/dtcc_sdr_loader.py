"""
DTCC PPD cross-currency basis swap loader.
==========================================

Reads the DTCC Public Price Dissemination (PPD) "cumulative rates" daily
CSV files produced under the CFTC Part 43 real-time reporting regime,
filters to cross-currency basis swap trades, and aggregates them into a
notional-weighted median basis (in bps) per (pair, tenor) bucket.

Source endpoint (public S3, no auth)::

    https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/
        CFTC_CUMULATIVE_RATES_{YYYY}_{MM}_{DD}.zip

ZIPs are cached under ``data/dtcc_sdr_cache/`` after the first download and
read directly on subsequent calls.

Schema quirks handled
---------------------
* ``Product name`` is empty for every row in the rates file -- we classify
  via ``UPI Underlier Name`` (e.g. ``EUR-EuroSTR-OIS Compound vs USD-SOFR-OIS
  Compound``) using the leg-1/leg-2 RFR keyword tables below.
* ``Spread-Leg 1/2`` is reported in **decimal**, ``Spread notation = 3``.
  ``-0.0002375`` is therefore -2.375 bps; we multiply by 1e4 to convert.
* Sign convention: the basis sits on the **non-USD leg**; the USD leg
  spread is ~0.  We pick whichever side has a non-trivial absolute spread
  and convert it.  This matches CME's XEURBI convention (basis on the
  non-USD leg).
* Notional capping appears in two forms in the DTCC rates file:
  (a) a trailing ``+`` (e.g. ``250,000,000+``), and (b) a 20-digit
  sentinel ``99,999,999,999,999,999,999.99999``.  Both indicate the
  trade was over the CFTC Part 43 block threshold; we replace the
  reported amount with the standard $250M Super-Major cap so the row
  contributes a sensible weight to the median.  Zero-notional rows are
  treated as missing.
* Notionals are comma-formatted strings.  We strip commas before float
  cast.
* Action-chain dedup: each xccy trade can have NEWT, then any number of
  MODI/CORR, optionally TERM/CANCEL/EROR.  We group by
  ``Original Dissemination Identifier`` (falling back to the row's own
  ``Dissemination Identifier`` when it IS the original NEWT), drop chains
  that end in ``CANCEL``/``EROR``, and keep the latest surviving event
  per chain.

Public API
----------
``fetch_xccy_basis_curve(valuation_date, pair, window_bd=5, rfr_only=True)``
    Returns a DataFrame with columns
    ``["pair", "tenor_label", "tenor_years", "basis_bps", "n_trades",
       "weighted_notional_usd", "source"]`` for the standard tenors
    {"3M", "6M", "1Y", "2Y", "5Y"}.
``fetch_dtcc_rates_csv(date_str)``
    Low-level: returns the full Rates CSV for ``date_str`` as a DataFrame.
    Downloads the ZIP if not cached.  Empty DataFrame if the file is
    missing on the upstream (e.g. weekend / holiday).
"""

from __future__ import annotations

import datetime as _dt
import io
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from shared.date_utils import is_business_day

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "dtcc_sdr_cache"

DTCC_URL_TEMPLATE = (
    "https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/"
    "CFTC_CUMULATIVE_RATES_{y}_{m:02d}_{d:02d}.zip"
)

# Standard tenor buckets in days (inclusive on both ends).
TENOR_BUCKETS: dict[str, tuple[int, int, float]] = {
    # label : (min_days, max_days, tenor_years_for_output)
    "3M": (75, 105, 0.25),
    "6M": (165, 195, 0.50),
    "1Y": (330, 390, 1.00),
    "2Y": (700, 790, 2.00),
    "5Y": (1700, 1900, 5.00),
}

# RFR / xccy underlier strings we accept by default.  These keywords are
# matched substring-wise against ``UPI Underlier Name`` (case-insensitive).
# When ``rfr_only=False`` we also accept the BBSW / HIBOR / NIBOR / BKBM
# IBOR-style underliers so AUD/USD coverage doesn't go to zero.
_RFR_TOKENS = {
    "USD": ("SOFR",),
    "EUR": ("EUROSTR", "ESTR"),
    "GBP": ("SONIA",),
    "JPY": ("TONA", "TONAR"),
    "AUD": ("AONIA",),
}
_IBOR_FALLBACK_TOKENS = {
    # Used only when rfr_only=False.
    "AUD": ("BBSW",),
    "GBP": ("LIBOR",),  # typically off the table post-cessation but defensive
    "JPY": ("LIBOR",),
}

# Pair -> (foreign_ccy, domestic_ccy).  Domestic is always USD here.
_PAIR_LEGS: dict[str, tuple[str, str]] = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "JPY/USD": ("JPY", "USD"),
    "AUD/USD": ("AUD", "USD"),
}

# Output columns for the public basis curve helper.
_BASIS_CURVE_COLS = [
    "pair",
    "tenor_label",
    "tenor_years",
    "basis_bps",
    "n_trades",
    "weighted_notional_usd",
    "source",
]


# ---------------------------------------------------------------------------
# Low-level: download + read one day's Rates CSV
# ---------------------------------------------------------------------------
def _zip_url_for(date: _dt.date) -> str:
    return DTCC_URL_TEMPLATE.format(y=date.year, m=date.month, d=date.day)


def _zip_cache_path_for(date: _dt.date) -> Path:
    return CACHE_DIR / f"CFTC_CUMULATIVE_RATES_{date.year}_{date.month:02d}_{date.day:02d}.zip"


def _download_zip(date: _dt.date, *, timeout: float = 30.0) -> Path | None:
    """Download the DTCC rates ZIP for ``date`` into the cache.

    Returns the cached path on success, or ``None`` if the upstream returns
    404 (e.g. weekend / holiday / not yet published).
    """
    cache_path = _zip_cache_path_for(date)
    if cache_path.exists():
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = _zip_url_for(date)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        # DTCC returns 404 for past dates that never had a file (typically
        # weekends/holidays it doesn't publish for) and 403 for future
        # dates whose key doesn't exist yet in the S3 bucket.  Both are
        # "file not published"; everything else is a genuine error.
        if exc.code in (403, 404):
            logger.info("DTCC rates ZIP not published for %s (HTTP %d)", date, exc.code)
            return None
        logger.warning("DTCC download failed for %s: %s", date, exc)
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("DTCC download network error for %s: %s", date, exc)
        return None

    cache_path.write_bytes(data)
    return cache_path


def fetch_dtcc_rates_csv(date_str: str) -> pd.DataFrame:
    """Read one day's CFTC Rates cumulative CSV from cache (download if needed).

    Returns an empty DataFrame if the file isn't published for ``date_str``
    (weekend, holiday, or future date).  All columns are read as strings;
    callers should coerce numeric fields explicitly so we don't silently
    lose the comma-formatting / capping markers.
    """
    date = _dt.date.fromisoformat(date_str)
    cache_path = _download_zip(date)
    if cache_path is None:
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(cache_path) as zf:
            # The archive contains a single CSV; pick the first .csv member.
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                logger.warning("No CSV inside %s", cache_path)
                return pd.DataFrame()
            with zf.open(csv_names[0]) as fh:
                df = pd.read_csv(fh, dtype=str, low_memory=False, na_filter=False)
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("Failed to read %s: %s", cache_path, exc)
        return pd.DataFrame()

    return df


# ---------------------------------------------------------------------------
# Schema-quirk helpers
# ---------------------------------------------------------------------------
# Standard CFTC Part 43 Super-Major cap for IRS / xccy basis trades.
# Used to give a sensible non-zero weight to capped rows when notional-
# weighting; the true size is unknown but is at least this value.
_BLOCK_CAP_USD = 250_000_000.0
# Anything at or above this magnitude is treated as the DTCC sentinel
# (``99,999,999,999,999,999,999.99999`` ~= 1e20) rather than a real notional.
_NOTIONAL_SENTINEL_THRESHOLD = 1e15


def _parse_notional(value: str) -> tuple[float, bool]:
    """Parse a comma-formatted notional with cap markers.

    Returns ``(amount, capped)``:
    * ``amount`` is the parsed dollar notional (or the standard block cap
      when the row is capped).
    * ``capped=True`` means the displayed value was a cap indicator
      (either trailing ``+`` or the 20-digit sentinel) rather than the
      true trade size.

    Zero values and empty / unparseable strings -> ``(NaN, False)``.
    """
    if value is None:
        return float("nan"), False
    s = str(value).strip()
    if not s:
        return float("nan"), False
    capped = s.endswith("+")
    if capped:
        s = s[:-1]
    s = s.replace(",", "")
    try:
        amt = float(s)
    except ValueError:
        return float("nan"), capped
    if amt == 0.0:
        return float("nan"), False
    if amt >= _NOTIONAL_SENTINEL_THRESHOLD:
        # 20-digit DTCC cap sentinel; substitute the standard block cap.
        return _BLOCK_CAP_USD, True
    if capped:
        return amt, True
    return amt, False


def _parse_spread_decimal(value: str) -> float:
    """Parse a decimal spread (notation=3).  Empty -> NaN.

    The DTCC file reports spreads as decimals: ``-0.0002375`` means
    -2.375 bps when ``Spread notation = 3``.  We don't read the notation
    column because every row in the rates file uses notation 3; if that
    ever changes the caller should branch.
    """
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _classify_pair(
    underlier: str,
    leg1_ccy: str,
    leg2_ccy: str,
    foreign: str,
    rfr_only: bool,
) -> bool:
    """Return True iff this row is a ``foreign``/USD xccy basis swap.

    Matches on:
    1. The two notional currencies must be {foreign, "USD"} (in either order).
    2. The UPI underlier name must reference the foreign-leg RFR (or an
       IBOR fallback when ``rfr_only=False``) AND USD SOFR on the other side.
    """
    legs = {leg1_ccy.upper(), leg2_ccy.upper()}
    if legs != {foreign.upper(), "USD"}:
        return False

    name = underlier.upper()
    if "VS" not in name:
        return False
    # Must mention USD SOFR somewhere.
    if not any(tok in name for tok in _RFR_TOKENS["USD"]):
        return False
    # Must mention the foreign-leg RFR.
    accept = list(_RFR_TOKENS.get(foreign.upper(), ()))
    if not rfr_only:
        accept.extend(_IBOR_FALLBACK_TOKENS.get(foreign.upper(), ()))
    if not any(tok in name for tok in accept):
        return False
    return True


def _basis_bps_from_legs(
    spread_leg1_dec: float,
    spread_leg2_dec: float,
    leg1_ccy: str,
    leg2_ccy: str,
    foreign: str,
) -> float:
    """Pull the basis (in bps) off the non-USD leg.

    Convention: USD leg spread is ~0 by construction.  We take whichever
    leg is the foreign currency and convert decimal -> bps.
    """
    f = foreign.upper()
    if leg1_ccy.upper() == f:
        return spread_leg1_dec * 1e4
    if leg2_ccy.upper() == f:
        return spread_leg2_dec * 1e4
    # Neither leg matches -- shouldn't happen post _classify_pair, but be
    # defensive.
    return float("nan")


def _dedupe_action_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each NEWT/MODI/... chain to its latest surviving event.

    A "chain" is the set of rows sharing the same originating
    ``Dissemination Identifier`` (i.e. the NEWT's ID).  For a row that IS
    the original NEWT, ``Original Dissemination Identifier`` is empty/NaN;
    we use that row's own ``Dissemination Identifier`` as the chain key.

    Chains whose latest event is ``CANCEL``/``EROR`` are dropped (the trade
    was retracted).  For surviving chains we keep the latest event, sorted
    by ``Event timestamp``.
    """
    out = df.copy()
    orig = out["Original Dissemination Identifier"].fillna("").astype(str).str.strip()
    own = out["Dissemination Identifier"].fillna("").astype(str).str.strip()
    out["__chain"] = np.where(orig == "", own, orig)

    # Drop any rows in chains that ever resolved to CANCEL/EROR.
    bad_mask = out["Action type"].isin({"CANCEL", "EROR"})
    bad_chains = set(out.loc[bad_mask, "__chain"].unique())
    out = out[~out["__chain"].isin(bad_chains)]

    # Sort by event timestamp, keep last per chain.
    out["__ts"] = pd.to_datetime(out["Event timestamp"], errors="coerce", utc=True)
    out = out.sort_values("__ts").drop_duplicates("__chain", keep="last")
    return out.drop(columns=["__chain", "__ts"])


# ---------------------------------------------------------------------------
# Date-window helpers
# ---------------------------------------------------------------------------
def _business_day_window(center: _dt.date, half_width: int) -> list[_dt.date]:
    """Return a sorted list of business days in ``[center-h, center+h]``.

    The center date is included (forced into the window even if it's a
    non-business day -- the upstream simply won't have a file, which we
    handle gracefully).  ``half_width`` is the number of business days on
    each side of ``center``.
    """
    if half_width < 0:
        raise ValueError("half_width must be >= 0")

    # Walk backward h business days, then forward h business days.
    days: list[_dt.date] = []

    # Backward.
    cur = center
    back: list[_dt.date] = []
    count = 0
    while count < half_width:
        cur = cur - _dt.timedelta(days=1)
        if is_business_day(cur):
            back.append(cur)
            count += 1
    days.extend(reversed(back))

    days.append(center)

    # Forward.
    cur = center
    count = 0
    while count < half_width:
        cur = cur + _dt.timedelta(days=1)
        if is_business_day(cur):
            days.append(cur)
            count += 1
    return days


# ---------------------------------------------------------------------------
# Public: per-tenor xccy basis aggregation
# ---------------------------------------------------------------------------
def fetch_xccy_basis_curve(
    valuation_date: str,
    pair: str,
    window_bd: int = 5,
    rfr_only: bool = True,
) -> pd.DataFrame:
    """Aggregate DTCC PPD prints into a basis curve for ``pair``.

    Parameters
    ----------
    valuation_date : str
        ISO date (``"YYYY-MM-DD"``).  Centre of the rolling window.
    pair : str
        One of ``{"EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"}``.
    window_bd : int, default 5
        Half-width of the business-day window.  ``window_bd=0`` uses only
        the valuation date; ``window_bd=5`` uses ±5 BDs (roughly two
        weeks).  Wider windows cheaply backfill thin tenors at the cost
        of some staleness.
    rfr_only : bool, default True
        When ``True`` only RFR-vs-RFR underliers are kept (e.g. €STR vs
        SOFR).  When ``False`` we additionally accept IBOR-style
        underliers (AUD-BBSW vs USD-SOFR, etc.) so AUD/USD coverage isn't
        zero on a typical day -- callers should flag this in the README.

    Returns
    -------
    pd.DataFrame
        Columns: ``pair, tenor_label, tenor_years, basis_bps, n_trades,
        weighted_notional_usd, source``.  One row per standard tenor in
        ``TENOR_BUCKETS``.  ``basis_bps`` is the notional-weighted median
        across all surviving trades in the bucket; rows where the bucket
        is empty have ``basis_bps=NaN`` and ``n_trades=0`` so the caller
        can apply a CIP fallback.
    """
    if pair not in _PAIR_LEGS:
        raise ValueError(f"Unsupported pair {pair!r}; expected one of {list(_PAIR_LEGS)}")
    foreign, _ = _PAIR_LEGS[pair]

    center = _dt.date.fromisoformat(valuation_date)
    window = _business_day_window(center, window_bd)

    pieces: list[pd.DataFrame] = []
    for d in window:
        raw = fetch_dtcc_rates_csv(d.isoformat())
        if raw.empty:
            continue
        # Subset and coerce only the columns we need to keep memory low.
        cols = [
            "Dissemination Identifier",
            "Original Dissemination Identifier",
            "Action type",
            "Event timestamp",
            "Effective Date",
            "Expiration Date",
            "Notional amount-Leg 1",
            "Notional amount-Leg 2",
            "Notional currency-Leg 1",
            "Notional currency-Leg 2",
            "Spread-Leg 1",
            "Spread-Leg 2",
            "UPI Underlier Name",
        ]
        # Some early files might miss columns; reindex to be safe.
        sub = raw.reindex(columns=cols).copy()
        pieces.append(sub)

    if not pieces:
        return _empty_basis_curve(pair)

    raw_window = pd.concat(pieces, ignore_index=True)

    # Filter to xccy basis swaps for this pair.
    leg1 = raw_window["Notional currency-Leg 1"].fillna("").astype(str)
    leg2 = raw_window["Notional currency-Leg 2"].fillna("").astype(str)
    name = raw_window["UPI Underlier Name"].fillna("").astype(str)
    classify_mask = [
        _classify_pair(n, c1, c2, foreign, rfr_only)
        for n, c1, c2 in zip(name, leg1, leg2)
    ]
    xccy = raw_window[classify_mask].copy()
    if xccy.empty:
        return _empty_basis_curve(pair)

    # Dedup action chains (drops CANCEL/EROR and consolidates MODI/CORR/TERM
    # back to the latest surviving event per trade).
    xccy = _dedupe_action_chains(xccy)
    if xccy.empty:
        return _empty_basis_curve(pair)

    # Compute tenor (effective_date -> expiration_date in days).
    eff = pd.to_datetime(xccy["Effective Date"], errors="coerce")
    exp = pd.to_datetime(xccy["Expiration Date"], errors="coerce")
    xccy["__tenor_days"] = (exp - eff).dt.days

    # Compute basis bps off the non-USD leg.
    spreads_l1 = xccy["Spread-Leg 1"].apply(_parse_spread_decimal)
    spreads_l2 = xccy["Spread-Leg 2"].apply(_parse_spread_decimal)
    xccy["__basis_bps"] = [
        _basis_bps_from_legs(s1, s2, c1, c2, foreign)
        for s1, s2, c1, c2 in zip(spreads_l1, spreads_l2, xccy["Notional currency-Leg 1"], xccy["Notional currency-Leg 2"])
    ]

    # Notional in USD: prefer the leg already in USD; if both legs are in
    # USD (shouldn't happen for xccy) take leg 1.  Cap at threshold when
    # the row is capped.
    n1 = xccy["Notional amount-Leg 1"].apply(_parse_notional)
    n2 = xccy["Notional amount-Leg 2"].apply(_parse_notional)
    xccy["__n1_amt"] = [v[0] for v in n1]
    xccy["__n1_capped"] = [v[1] for v in n1]
    xccy["__n2_amt"] = [v[0] for v in n2]
    xccy["__n2_capped"] = [v[1] for v in n2]
    # USD notional pick:
    leg1_is_usd = xccy["Notional currency-Leg 1"].str.upper() == "USD"
    xccy["__notional_usd"] = np.where(leg1_is_usd, xccy["__n1_amt"], xccy["__n2_amt"])
    xccy["__notional_capped"] = np.where(leg1_is_usd, xccy["__n1_capped"], xccy["__n2_capped"])

    xccy = xccy.dropna(subset=["__tenor_days", "__basis_bps", "__notional_usd"])

    # Bucket by tenor and aggregate.
    rows: list[dict] = []
    for label, (lo, hi, tenor_years) in TENOR_BUCKETS.items():
        bucket = xccy[
            (xccy["__tenor_days"] >= lo) & (xccy["__tenor_days"] <= hi)
        ]
        if bucket.empty:
            rows.append({
                "pair": pair,
                "tenor_label": label,
                "tenor_years": tenor_years,
                "basis_bps": float("nan"),
                "n_trades": 0,
                "weighted_notional_usd": 0.0,
                "source": "dtcc_sdr_empty",
            })
            continue

        weights = bucket["__notional_usd"].astype(float).to_numpy()
        # Capped rows: cap weight at the threshold value.  Since the
        # displayed notional already IS the cap value when capped=True,
        # this is effectively a no-op here -- we keep the structure
        # explicit so a future change (e.g. a different cap rule) is a
        # one-liner.
        # Per CFTC Appendix A the value shown when capped is the cap
        # itself, so weights[i] = bucket['__notional_usd'].iloc[i] already
        # reflects the cap.  No further adjustment needed.
        basis = bucket["__basis_bps"].astype(float).to_numpy()

        rows.append({
            "pair": pair,
            "tenor_label": label,
            "tenor_years": tenor_years,
            "basis_bps": _weighted_median(basis, weights),
            "n_trades": int(len(bucket)),
            "weighted_notional_usd": float(weights.sum()),
            "source": "dtcc_sdr",
        })

    return pd.DataFrame(rows, columns=_BASIS_CURVE_COLS)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Notional-weighted median of ``values``.

    Drops NaNs in either array; returns NaN if no valid points.
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[mask]
    w = weights[mask]
    if v.size == 0:
        return float("nan")
    if v.size == 1:
        return float(v[0])
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    idx = int(np.searchsorted(cum, cutoff))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _empty_basis_curve(pair: str) -> pd.DataFrame:
    rows = [
        {
            "pair": pair,
            "tenor_label": label,
            "tenor_years": ty,
            "basis_bps": float("nan"),
            "n_trades": 0,
            "weighted_notional_usd": 0.0,
            "source": "dtcc_sdr_empty",
        }
        for label, (_, _, ty) in TENOR_BUCKETS.items()
    ]
    return pd.DataFrame(rows, columns=_BASIS_CURVE_COLS)
