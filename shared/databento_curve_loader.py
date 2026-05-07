"""
Databento curve loader (cache-only, read-only).
================================================

This module reads OHLCV-1d parquet cache files that were already fetched by
the live Databento pipeline (see ``shared/databento_loader.py``) and reshapes
them into curve-friendly DataFrames suitable for downstream cross-currency
basis derivation and SOFR / SONIA / ESTR OIS bootstrapping.

It does **not** make any live Databento API calls.  The live API path lives
in ``shared/databento_loader.py`` (FX MBP-10) and ``prefetch_data.py``
(OHLCV-1d strips).  This module is purely a parquet -> DataFrame
transformer.

Public functions
----------------
fetch_back_month_fx_curve(ticker, date_str, n_contracts=4) -> pd.DataFrame
    Returns the CME FX futures curve (front + back months) on one valuation
    date with columns ``["contract", "expiry_years", "price"]``.

fetch_sofr_strip(date_str, n_contracts=8) -> pd.DataFrame
    Returns the 3M SOFR (SR3) futures strip.

fetch_sr1_strip(date_str, n_contracts=7) -> pd.DataFrame
    Returns the 1M SOFR (SR1) futures strip (monthly cycle).

fetch_sonia_so3_strip(date_str, n_contracts=8) -> pd.DataFrame
    Returns the 3M SONIA (SO3) futures strip (ICE, quarterly cycle).

fetch_sonia_soa_strip(date_str, n_contracts=6) -> pd.DataFrame
    Returns the 1M SONIA (SOA) futures strip (ICE, monthly cycle).

fetch_estr_fst3_strip(date_str, n_contracts=8) -> pd.DataFrame
    Returns the 3M ESTR (FST3) futures strip (Eurex, quarterly cycle).

fetch_estr_femp_strip(date_str, n_contracts=6) -> pd.DataFrame
    Returns the ECB-dated ESTR (FEMP) futures strip (Eurex, ~6 week
    maintenance-period cycle).

Each strip fetcher returns columns
``["contract", "expiry_years", "price", "implied_rate"]`` where
``implied_rate`` is the decimal rate ``(100 - price) / 100`` and
``expiry_years`` is ACT/365.25 time from the valuation date to the
contract reference-period start.

If the expected cache file is missing, all functions return an empty
DataFrame with the right column schema rather than raising.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "databento_cache"

# Quarterly month code mapping used by CME FX and SR3 futures.
# H = March, M = June, U = September, Z = December.
QUARTER_MONTHS = [(3, "H"), (6, "M"), (9, "U"), (12, "Z")]

# Output column schemas (used both for non-empty and empty returns).
_FX_COLS = ["contract", "expiry_years", "price"]
_SOFR_COLS = ["contract", "expiry_years", "price", "implied_rate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _month_code_to_month(code: str) -> int:
    """Map a quarterly month code letter to the calendar month (1-12)."""
    for month, letter in QUARTER_MONTHS:
        if letter == code:
            return month
    raise ValueError(f"Unknown quarterly month code: {code!r}")


def _resolve_year(year_digit: int, val_date: _dt.date) -> int:
    """Resolve a single-digit year from a Databento contract symbol.

    The contract year is given as a single digit (e.g. ``6EM6.c.0`` -> 6).
    We anchor this to the decade of ``val_date``: if the resulting year is
    more than 5 years before ``val_date``, roll forward by 10 years so a
    ``6`` on a 2026-04-28 valuation maps to 2026 rather than 2016.
    """
    decade_base = (val_date.year // 10) * 10
    candidate = decade_base + year_digit
    if candidate < val_date.year - 5:
        candidate += 10
    return candidate


def _third_wednesday(year: int, month: int) -> _dt.date:
    """Return the third Wednesday of ``year``-``month``.

    CME FX quarterly futures expire on the third Wednesday of the contract
    month (the day before the 3rd-Friday LIBOR/IMM convention is typically
    used for the rate complex; FX uses 3rd Wed).
    """
    first = _dt.date(year, month, 1)
    # weekday(): Monday=0, ..., Wednesday=2.
    offset = (2 - first.weekday()) % 7
    first_wed = first + _dt.timedelta(days=offset)
    return first_wed + _dt.timedelta(days=14)


def _parse_fx_symbol(
    symbol: str,
    ticker: str,
    val_date: "_dt.date | None" = None,
) -> tuple[str, int] | None:
    """Parse a Databento FX continuous symbol into (month_code, year_digit).

    Two layouts supported:

    1. ``{ticker}{month_code}{year_digit}[.c.{n}]`` — explicit form with
       embedded month code, e.g. ``6EM6.c.0`` -> ("M", 6).
    2. ``{ticker}.c.{n}`` — bare-root continuous form (the Databento default
       for FX cache writes), e.g. ``6E.c.0``.  Resolved by walking ``n``
       quarterly contracts forward from the front quarter on/after
       ``val_date``.  When ``val_date`` is ``None`` we cannot resolve a bare
       root and return ``None``.

    Returns ``None`` on any layout we don't recognise.
    """
    if not symbol.startswith(ticker):
        return None
    body = symbol[len(ticker):]

    # Bare-root continuous form: "{ticker}.c.{n}".
    if body.startswith(".c."):
        if val_date is None:
            return None
        try:
            n = int(body.split(".c.", 1)[1])
        except (ValueError, IndexError):
            return None
        front_code, front_year = _front_quarter_on_or_after(val_date)
        month_code, year = _step_quarters(front_code, front_year, n)
        return month_code, year % 10

    # Explicit form: "{ticker}{month_code}{year_digit}[.c.{n}]".
    if "." in body:
        body = body.split(".", 1)[0]
    if len(body) < 2:
        return None
    month_code = body[0]
    year_chars = body[1:]
    if not year_chars.isdigit():
        return None
    # Use just the last digit (Databento uses single-digit year for these
    # quarterly contracts).
    year_digit = int(year_chars[-1])
    if month_code not in {letter for _, letter in QUARTER_MONTHS}:
        return None
    return month_code, year_digit


def _front_quarter_on_or_after(val_date: _dt.date) -> tuple[str, int]:
    """Return ``(month_code, year)`` of the first quarterly month on or after
    ``val_date``.

    The CME quarterly cycle is March (H), June (M), September (U), December
    (Z).  Used to resolve bare-root continuous symbols (e.g. ``SR3.c.0``)
    where the contract month/year is implicit and indexed off the valuation
    date's front contract.
    """
    for month, code in QUARTER_MONTHS:
        if val_date.month <= month:
            return code, val_date.year
    # Past December -> roll into next year's March.
    return "H", val_date.year + 1


def _step_quarters(month_code: str, year: int, n: int) -> tuple[str, int]:
    """Advance ``(month_code, year)`` forward by ``n`` quarterly steps.

    Used to walk the continuous-symbol index: ``SR3.c.0`` -> front quarter,
    ``SR3.c.1`` -> next quarter, etc.
    """
    quarter_codes = [code for _, code in QUARTER_MONTHS]
    idx = quarter_codes.index(month_code)
    total = idx + n
    new_year = year + (total // 4)
    new_code = quarter_codes[total % 4]
    return new_code, new_year


# ---------------------------------------------------------------------------
# Monthly + ECB-cycle helpers (used by SR1, SOA, FEMP fetchers)
# ---------------------------------------------------------------------------
# Full CME/ICE monthly month-code map (no I or L; J=April, K=May, etc.).
MONTH_CODES = [
    (1, "F"), (2, "G"), (3, "H"), (4, "J"), (5, "K"), (6, "M"),
    (7, "N"), (8, "Q"), (9, "U"), (10, "V"), (11, "X"), (12, "Z"),
]


def _front_month_on_or_after(val_date: _dt.date) -> tuple[int, int]:
    """Return ``(month, year)`` of the first calendar month >= ``val_date``.

    Used to anchor monthly continuous-symbol indices like ``SR1.c.0``.
    A valuation in mid-month resolves to that same calendar month: e.g.
    2026-04-28 -> (4, 2026).
    """
    return val_date.month, val_date.year


def _step_months(month: int, year: int, n: int) -> tuple[int, int]:
    """Advance ``(month, year)`` forward by ``n`` calendar-month steps."""
    total = (month - 1) + n
    new_year = year + total // 12
    new_month = (total % 12) + 1
    return new_month, new_year


def _monthly_imm_date(month: int, year: int) -> _dt.date:
    """IMM date for a monthly RFR contract = 3rd Wednesday of the month.

    SR1, SOA all settle to the compounded RFR over a calendar-month
    reference period that nominally starts on the 3rd Wednesday of the
    contract month.  This matches the existing ``_third_wednesday`` helper
    used for SR3.
    """
    return _third_wednesday(year, month)


# Approximate ECB maintenance-period length: the ECB Governing Council meets
# roughly every 6 weeks, so FEMP "ECB-dated" contracts also cycle with a
# ~6-week period.  Real maintenance-period dates are published by the ECB
# and would need to be tabulated for an exact answer; spec calls this out
# as an explicit approximation.
_FEMP_PERIOD_DAYS = 42


def _femp_period_start(val_date: _dt.date, n: int) -> _dt.date:
    """Approximate start of the n-th forward FEMP ECB-maintenance period.

    FEMP futures settle to compounded ESTR over an ECB maintenance period
    (typically ~6 weeks between ECB meetings).  We do not have the real
    ECB calendar in this module, so we approximate the n-th forward period
    as ``val_date + (n + 1) * 42`` days; this is good enough for the data
    layer's purpose (sorting and rough expiry-year computation) and is
    explicitly flagged in the public fetcher's docstring.
    """
    return val_date + _dt.timedelta(days=(n + 1) * _FEMP_PERIOD_DAYS)


def _parse_continuous_index(symbol: str, root: str) -> int | None:
    """Parse a bare-root continuous symbol like ``"SR1.c.3"`` -> ``3``.

    Returns ``None`` if the symbol does not match ``{root}.c.{int}``.
    """
    if not symbol.startswith(root):
        return None
    body = symbol[len(root):]
    if not body.startswith("."):
        return None
    parts = body[1:].split(".")
    if len(parts) != 2 or parts[0] != "c" or not parts[1].isdigit():
        return None
    return int(parts[1])


def _parse_sr3_symbol(
    symbol: str,
    val_date: _dt.date | None = None,
) -> tuple[str, int] | None:
    """Parse a SR3 SOFR futures symbol -> ``(month_code, year_digit)``.

    Two symbol layouts are supported:

    * Explicit quarterly contract, e.g. ``SR3M6.c.0`` or ``SR3M6`` ->
      ``("M", 6)``.  The ``.c.N`` continuous-roll suffix is stripped and
      the embedded month code + single year digit are used directly.

    * Bare-root continuous, e.g. ``SR3.c.0``, ``SR3.c.1``, ... where the
      contract month/year is implicit.  ``.c.N`` is interpreted as the
      N-th quarterly contract starting from the front quarter on or after
      ``val_date`` (so ``SR3.c.0`` is the front, ``SR3.c.1`` the next
      quarter, etc.).  This layout requires ``val_date`` to be supplied;
      we return ``None`` otherwise.

    The function returns the year as a single digit (``year_digit`` =
    ``year % 10``) to match the existing downstream contract: callers
    pass it through ``_resolve_year`` to recover the four-digit year and
    then to ``_third_wednesday`` to obtain the contract IMM date (the
    start of the SR3 3-month reference period).  Returns ``None`` if the
    symbol does not start with ``SR3`` or otherwise fails to parse.
    """
    if not symbol.startswith("SR3"):
        return None
    body = symbol[len("SR3"):]
    # Drop ".c.N" continuous-roll suffix if present.
    suffix_idx = body.find(".")
    head = body if suffix_idx < 0 else body[:suffix_idx]

    if head:
        # Explicit form: head looks like "M6" (month code + single year digit).
        if len(head) < 2:
            return None
        month_code = head[0]
        year_chars = head[1:]
        if not year_chars.isdigit():
            return None
        if month_code not in {letter for _, letter in QUARTER_MONTHS}:
            return None
        return month_code, int(year_chars[-1])

    # Bare-root form: "SR3.c.N".  Need val_date to resolve.
    if val_date is None:
        return None
    if suffix_idx < 0:
        # Just "SR3" with no suffix -- ambiguous without an index; treat as front.
        n = 0
    else:
        tail = body[suffix_idx + 1:]  # e.g. "c.0"
        # Expected form "c.N".
        parts = tail.split(".")
        if len(parts) != 2 or parts[0] != "c" or not parts[1].isdigit():
            return None
        n = int(parts[1])

    front_code, front_year = _front_quarter_on_or_after(val_date)
    month_code, year = _step_quarters(front_code, front_year, n)
    return month_code, year % 10


def _sr3_imm_date(month_code: str, year: int) -> _dt.date:
    """SR3 IMM date = start of the 3-month reference period.

    SR3 contracts reference SOFR over a 3-month period that starts on the
    contract IMM date and ends 3 months later.  The IMM date is the third
    Wednesday of the contract month (e.g. the M6 contract uses the third
    Wednesday of June 2026 = 2026-06-17 as the start of its reference
    period).  ``expiry_years`` for downstream bootstrappers is the time
    from valuation to this IMM date.
    """
    contract_month = _month_code_to_month(month_code)
    return _third_wednesday(year, contract_month)


def _empty_fx_df() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object" if c == "contract" else "float64")
                         for c in _FX_COLS})


def _empty_sofr_df() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object" if c == "contract" else "float64")
                         for c in _SOFR_COLS})


# ---------------------------------------------------------------------------
# Public: FX back-month curve
# ---------------------------------------------------------------------------
def fetch_back_month_fx_curve(
    ticker: str,
    date_str: str,
    n_contracts: int = 4,
) -> pd.DataFrame:
    """Read the CME FX back-month futures curve from the Databento cache.

    Parameters
    ----------
    ticker : str
        Generic CME FX root, e.g. ``"6E"`` (EUR/USD), ``"6B"`` (GBP/USD),
        ``"6J"`` (JPY/USD), ``"6A"`` (AUD/USD).
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 4
        Maximum number of contracts to return (front + back months in
        ascending expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price"]`` sorted ascending
        by ``expiry_years``.  Empty DataFrame (right schema) if the cache
        file is missing or unparsable.
    """
    path = CACHE_DIR / f"fx_back_month_curve_{date_str}.parquet"
    if not path.exists():
        logger.info("FX back-month cache miss: %s", path)
        return _empty_fx_df()

    try:
        raw = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return _empty_fx_df()

    if "symbol" not in raw.columns:
        logger.warning("Cache %s missing 'symbol' column", path)
        return _empty_fx_df()

    val_date = _dt.date.fromisoformat(date_str)

    # Filter to symbols that start with the requested root ticker.
    mask = raw["symbol"].astype(str).str.startswith(ticker)
    df = raw.loc[mask].copy()
    # Order rows by ts_event so "last per symbol" is well defined.
    if "ts_event" in df.columns and not df.empty:
        df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce", utc=True)
        df = df.sort_values("ts_event")

    # Primary: ohlcv-1d cache (actual daily close = last trade).
    last_close: pd.DataFrame
    if not df.empty and "close" in df.columns:
        last_close = df.groupby("symbol", as_index=False)["close"].last()
    elif not df.empty and "stat_type" in df.columns and "price" in df.columns:
        # Cache file holds statistics rows directly (legacy or settles-only path).
        settles = df[df["stat_type"] == 3].dropna(subset=["price"])
        last_close = settles.groupby("symbol", as_index=False)["price"].last()
        last_close = last_close.rename(columns={"price": "close"})
    else:
        last_close = pd.DataFrame(columns=["symbol", "close"])

    # Fallback: statistics settles (stat_type == 3) for any contract the
    # primary cache doesn't cover. Reads `fx_back_month_settles_{date}.parquet`
    # if present and unions only the missing symbols.
    settles_path = CACHE_DIR / f"fx_back_month_settles_{date_str}.parquet"
    if settles_path.exists():
        try:
            sraw = pd.read_parquet(settles_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", settles_path, exc)
            sraw = None
        if sraw is not None and "symbol" in sraw.columns and "stat_type" in sraw.columns and "price" in sraw.columns:
            sdf = sraw[sraw["symbol"].astype(str).str.startswith(ticker)].copy()
            if "ts_event" in sdf.columns and not sdf.empty:
                sdf["ts_event"] = pd.to_datetime(sdf["ts_event"], errors="coerce", utc=True)
                sdf = sdf.sort_values("ts_event")
            sdf = sdf[sdf["stat_type"] == 3].dropna(subset=["price"])
            if not sdf.empty:
                settles = sdf.groupby("symbol", as_index=False)["price"].last()
                settles = settles.rename(columns={"price": "close"})
                covered = set(last_close["symbol"].astype(str)) if not last_close.empty else set()
                fill = settles[~settles["symbol"].isin(covered)]
                if not fill.empty:
                    last_close = pd.concat([last_close, fill], ignore_index=True)

    if last_close.empty:
        return _empty_fx_df()

    rows: list[dict] = []
    for _, r in last_close.iterrows():
        symbol = str(r["symbol"])
        parsed = _parse_fx_symbol(symbol, ticker, val_date=val_date)
        if parsed is None:
            continue
        month_code, year_digit = parsed
        year = _resolve_year(year_digit, val_date)
        try:
            expiry = _third_wednesday(year, _month_code_to_month(month_code))
        except ValueError:
            continue
        expiry_years = (expiry - val_date).days / 365.25
        if expiry_years <= 0:
            # Skip already-expired contracts.
            continue
        rows.append({
            "contract": symbol,
            "expiry_years": float(expiry_years),
            "price": float(r["close"]),
        })

    if not rows:
        return _empty_fx_df()

    out = pd.DataFrame(rows, columns=_FX_COLS)
    out = out.sort_values("expiry_years").reset_index(drop=True)
    if n_contracts is not None and n_contracts > 0:
        out = out.head(n_contracts).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Public: SOFR (SR3) futures strip
# ---------------------------------------------------------------------------
def fetch_sofr_strip(date_str: str, n_contracts: int = 8) -> pd.DataFrame:
    """Read the SR3 SOFR futures strip from the Databento cache.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 8
        Maximum number of contracts to return (sorted by ascending expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price", "implied_rate"]``
        sorted ascending by ``expiry_years``.  ``implied_rate`` is the
        decimal rate ``(100 - price) / 100``.  Empty DataFrame (right
        schema) if the cache file is missing or unparsable.
    """
    path = CACHE_DIR / f"sofr_sr3_strip_1y_to_{date_str}.parquet"
    if not path.exists():
        logger.info("SOFR strip cache miss: %s", path)
        return _empty_sofr_df()

    try:
        raw = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return _empty_sofr_df()

    if "symbol" not in raw.columns or "close" not in raw.columns:
        logger.warning("Cache %s missing required columns", path)
        return _empty_sofr_df()

    val_date = _dt.date.fromisoformat(date_str)
    val_ts = pd.Timestamp(val_date)

    # Filter to SR3 contracts only.
    mask = raw["symbol"].astype(str).str.startswith("SR3")
    df = raw.loc[mask].copy()
    if df.empty:
        return _empty_sofr_df()

    # Restrict to bars dated on or before the valuation date so we get the
    # close "as of" date_str even when the cache spans a full year.
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce", utc=True)
        # Use a tz-naive comparison via the date component.
        df["bar_date"] = df["ts_event"].dt.tz_convert(None).dt.normalize() \
            if df["ts_event"].dt.tz is not None else df["ts_event"].dt.normalize()
        df = df[df["bar_date"] <= val_ts]
        df = df.sort_values("ts_event")
    if df.empty:
        return _empty_sofr_df()

    # Most recent close per symbol on or before the valuation date.
    last_close = df.groupby("symbol", as_index=False)["close"].last()

    rows: list[dict] = []
    for _, r in last_close.iterrows():
        symbol = str(r["symbol"])
        parsed = _parse_sr3_symbol(symbol, val_date=val_date)
        if parsed is None:
            continue
        month_code, year_digit = parsed
        year = _resolve_year(year_digit, val_date)
        try:
            imm = _sr3_imm_date(month_code, year)
        except ValueError:
            continue
        # expiry_years is time to the IMM date (start of the 3-month
        # SR3 reference period); the bootstrapper adds +0.25Y to obtain
        # ref_end and treats implied_rate as the simple forward rate
        # over [imm, imm + 3M].
        expiry_years = (imm - val_date).days / 365.25
        if expiry_years <= 0:
            continue
        price = float(r["close"])
        rows.append({
            "contract": symbol,
            "expiry_years": float(expiry_years),
            "price": price,
            "implied_rate": (100.0 - price) / 100.0,
        })

    if not rows:
        return _empty_sofr_df()

    out = pd.DataFrame(rows, columns=_SOFR_COLS)
    out = out.sort_values("expiry_years").reset_index(drop=True)
    if n_contracts is not None and n_contracts > 0:
        out = out.head(n_contracts).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Generic continuous-strip reader (used by SR1 / SO3 / SOA / FST3 / FEMP)
# ---------------------------------------------------------------------------
def _read_continuous_strip(
    cache_path: Path,
    root: str,
    val_date: _dt.date,
    period_start_fn,
    n_contracts: int,
) -> pd.DataFrame:
    """Generic reader for a bare-root continuous OHLCV-1d strip.

    Parameters
    ----------
    cache_path : Path
        Parquet file with at least ``symbol`` and ``close`` columns.
    root : str
        Bare-root symbol, e.g. ``"SR1"``.  Only rows whose ``symbol``
        column starts with ``root`` are kept, and only ``{root}.c.{int}``
        layouts are parsed.
    val_date : datetime.date
        Valuation date used to bound the "as-of" close and to drive
        ``period_start_fn``.
    period_start_fn : Callable[[int], datetime.date]
        Maps a continuous index ``n`` to the contract's reference-period
        start (i.e. the date used to compute ``expiry_years``).
    n_contracts : int
        Cap the output rows to the first ``n_contracts`` (sorted by
        ``expiry_years``).
    """
    if not cache_path.exists():
        logger.info("strip cache miss: %s", cache_path)
        return _empty_sofr_df()

    try:
        raw = pd.read_parquet(cache_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", cache_path, exc)
        return _empty_sofr_df()

    if "symbol" not in raw.columns or "close" not in raw.columns:
        logger.warning("Cache %s missing required columns", cache_path)
        return _empty_sofr_df()

    val_ts = pd.Timestamp(val_date)

    # Filter to root-prefixed contracts only.
    mask = raw["symbol"].astype(str).str.startswith(root)
    df = raw.loc[mask].copy()
    if df.empty:
        return _empty_sofr_df()

    # Restrict to bars dated on or before the valuation date.
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce", utc=True)
        df["bar_date"] = df["ts_event"].dt.tz_convert(None).dt.normalize() \
            if df["ts_event"].dt.tz is not None else df["ts_event"].dt.normalize()
        df = df[df["bar_date"] <= val_ts]
        df = df.sort_values("ts_event")
    if df.empty:
        return _empty_sofr_df()

    last_close = df.groupby("symbol", as_index=False)["close"].last()

    rows: list[dict] = []
    for _, r in last_close.iterrows():
        symbol = str(r["symbol"])
        n = _parse_continuous_index(symbol, root)
        if n is None:
            continue
        try:
            ref_start = period_start_fn(n)
        except (ValueError, OverflowError):
            continue
        expiry_years = (ref_start - val_date).days / 365.25
        if expiry_years <= 0:
            continue
        price = float(r["close"])
        rows.append({
            "contract": symbol,
            "expiry_years": float(expiry_years),
            "price": price,
            "implied_rate": (100.0 - price) / 100.0,
        })

    if not rows:
        return _empty_sofr_df()

    out = pd.DataFrame(rows, columns=_SOFR_COLS)
    out = out.sort_values("expiry_years").reset_index(drop=True)
    if n_contracts is not None and n_contracts > 0:
        out = out.head(n_contracts).reset_index(drop=True)
    return out


def _monthly_period_start_fn(val_date: _dt.date):
    """Build a ``n -> ref_start`` function for monthly RFR contracts.

    The n-th continuous contract is the n-th calendar month from (and
    including) ``val_date``'s month.  Reference period starts on the 3rd
    Wednesday of that month.  Used by SR1 (1M SOFR) and SOA (1M SONIA).
    """
    front_month, front_year = _front_month_on_or_after(val_date)

    def fn(n: int) -> _dt.date:
        m, y = _step_months(front_month, front_year, n)
        return _monthly_imm_date(m, y)

    return fn


def _quarterly_period_start_fn(val_date: _dt.date):
    """Build a ``n -> ref_start`` function for quarterly RFR contracts.

    Mirrors the SR3 convention: ref_start = 3rd Wednesday of the n-th
    forward quarterly month from ``val_date``'s front quarter.  Used by
    SO3 (3M SONIA) and FST3 (3M ESTR).
    """
    front_code, front_year = _front_quarter_on_or_after(val_date)

    def fn(n: int) -> _dt.date:
        code, year = _step_quarters(front_code, front_year, n)
        return _third_wednesday(year, _month_code_to_month(code))

    return fn


# ---------------------------------------------------------------------------
# Public: 1M SOFR (SR1) strip
# ---------------------------------------------------------------------------
def fetch_sr1_strip(date_str: str, n_contracts: int = 7) -> pd.DataFrame:
    """Read the 1M SOFR (SR1) futures strip from the Databento cache.

    SR1 is the CME 1-Month SOFR future: it settles to the arithmetic
    average of daily SOFR over a calendar-month reference period that
    starts on the 3rd Wednesday of the contract month.  The bare-root
    continuous symbols ``SR1.c.0``..``SR1.c.{n-1}`` step monthly forward
    from the front month on/after ``date_str``.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 7
        Maximum number of contracts to return (ascending in expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price", "implied_rate"]``.
    """
    val_date = _dt.date.fromisoformat(date_str)
    cache_path = CACHE_DIR / f"sofr_sr1_strip_1y_to_{date_str}.parquet"
    return _read_continuous_strip(
        cache_path,
        root="SR1",
        val_date=val_date,
        period_start_fn=_monthly_period_start_fn(val_date),
        n_contracts=n_contracts,
    )


# ---------------------------------------------------------------------------
# Public: 3M SONIA (SO3) strip
# ---------------------------------------------------------------------------
def fetch_sonia_so3_strip(date_str: str, n_contracts: int = 8) -> pd.DataFrame:
    """Read the 3M SONIA (SO3) futures strip from the Databento cache.

    SO3 is the ICE 3-Month SONIA future: it settles to compounded daily
    SONIA over a 3-month reference period starting on the 3rd Wednesday
    of the contract month (March/June/September/December cycle).  Bare-
    root continuous symbols ``SO3.c.0``..``SO3.c.{n-1}`` step quarterly
    forward from the front quarter on/after ``date_str``.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 8
        Maximum number of contracts to return (ascending in expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price", "implied_rate"]``.
    """
    val_date = _dt.date.fromisoformat(date_str)
    cache_path = CACHE_DIR / f"sonia_so3_strip_1y_to_{date_str}.parquet"
    return _read_continuous_strip(
        cache_path,
        root="SO3",
        val_date=val_date,
        period_start_fn=_quarterly_period_start_fn(val_date),
        n_contracts=n_contracts,
    )


# ---------------------------------------------------------------------------
# Public: 1M SONIA (SOA) strip
# ---------------------------------------------------------------------------
def fetch_sonia_soa_strip(date_str: str, n_contracts: int = 6) -> pd.DataFrame:
    """Read the 1M SONIA (SOA) futures strip from the Databento cache.

    SOA is the ICE 1-Month SONIA future, settling to compounded SONIA
    over a calendar-month period starting on the 3rd Wednesday of the
    contract month.  Bare-root continuous symbols ``SOA.c.0``..
    ``SOA.c.{n-1}`` step monthly forward from the front month on/after
    ``date_str``.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 6
        Maximum number of contracts to return (ascending in expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price", "implied_rate"]``.
    """
    val_date = _dt.date.fromisoformat(date_str)
    cache_path = CACHE_DIR / f"sonia_soa_strip_1y_to_{date_str}.parquet"
    return _read_continuous_strip(
        cache_path,
        root="SOA",
        val_date=val_date,
        period_start_fn=_monthly_period_start_fn(val_date),
        n_contracts=n_contracts,
    )


# ---------------------------------------------------------------------------
# Public: 3M ESTR (FST3) strip
# ---------------------------------------------------------------------------
def fetch_estr_fst3_strip(date_str: str, n_contracts: int = 8) -> pd.DataFrame:
    """Read the 3M ESTR (FST3) futures strip from the Databento cache.

    FST3 is the Eurex 3-Month ESTR future, settling to compounded ESTR
    over a 3-month reference period starting on the 3rd Wednesday of the
    contract month (March/June/September/December cycle).  Bare-root
    continuous symbols ``FST3.c.0``..``FST3.c.{n-1}`` step quarterly
    forward from the front quarter on/after ``date_str``.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 8
        Maximum number of contracts to return (ascending in expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price", "implied_rate"]``.
    """
    val_date = _dt.date.fromisoformat(date_str)
    cache_path = CACHE_DIR / f"estr_fst3_strip_1y_to_{date_str}.parquet"
    return _read_continuous_strip(
        cache_path,
        root="FST3",
        val_date=val_date,
        period_start_fn=_quarterly_period_start_fn(val_date),
        n_contracts=n_contracts,
    )


# ---------------------------------------------------------------------------
# Public: ECB-dated ESTR (FEMP) strip
# ---------------------------------------------------------------------------
def fetch_estr_femp_strip(date_str: str, n_contracts: int = 6) -> pd.DataFrame:
    """Read the ECB-dated ESTR (FEMP) futures strip from the Databento cache.

    FEMP is the Eurex "MPC-dated" ESTR future: each contract settles to
    compounded ESTR over a single ECB maintenance period (the window
    between two consecutive ECB Governing Council monetary-policy
    decisions).

    The exact maintenance-period start dates are published by the ECB
    and are not encoded in this module; we approximate the n-th forward
    period start as ``val_date + (n + 1) * 42 days`` (~6 weeks per ECB
    cycle), which is the right scale for sorting and rough expiry-year
    computation in the data layer.  Downstream consumers that need
    exact dates should override or supplement this with the published
    ECB calendar.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).
    n_contracts : int, default 6
        Maximum number of contracts to return (ascending in expiry).

    Returns
    -------
    pd.DataFrame
        Columns ``["contract", "expiry_years", "price", "implied_rate"]``.
    """
    val_date = _dt.date.fromisoformat(date_str)
    cache_path = CACHE_DIR / f"estr_femp_strip_1y_to_{date_str}.parquet"

    def femp_start(n: int) -> _dt.date:
        return _femp_period_start(val_date, n)

    return _read_continuous_strip(
        cache_path,
        root="FEMP",
        val_date=val_date,
        period_start_fn=femp_start,
        n_contracts=n_contracts,
    )
