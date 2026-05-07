"""
Exchange-direct RFR-futures loader (JPX TONA + ASX IB).
=======================================================

Databento does not carry JPX (3M TONA Futures) or ASX (30-Day Interbank Cash
Rate Futures), so we go direct to the exchange websites.  Both endpoints are
free and unauthenticated.

The output schema is intentionally aligned with
``shared/databento_curve_loader.py::fetch_sofr_strip`` so downstream
bootstrappers can consume any of the three RFR feeds (SOFR, TONA, AONIA-IB)
through one code path.

Public API
----------
fetch_tona_strip(date_str) -> pd.DataFrame
    OSE 3M TONA futures settlement strip.  Columns:
        ['contract', 'expiry_years', 'price', 'implied_rate']
    where ``expiry_years`` is the time from the valuation date to the
    contract's last trading day (the business day before the third
    Wednesday of the month three months after the listed contract month
    on the JPX feed), and ``implied_rate = (100 - settlement_price) / 100``.

fetch_aonia_ib_strip(date_str) -> pd.DataFrame
    ASX 30-Day Interbank Cash Rate Futures settlement strip.  Same column
    schema; ``expiry_years`` is the time from valuation to contract
    month-end (the IB future averages the cash rate over a single calendar
    month).

Both functions:
  * cache the parsed strip as parquet under
    ``data/exchange_rfr_cache/`` for 24h, and
  * return an EMPTY DataFrame with the right schema (NEVER raise) on any
    fetch / parse failure -- letting downstream code gracefully degrade.

Source try-chain
----------------
TONA (in order):
  1. JPX **daily-settlement CSV** linked off
     ``markets/derivatives/settlement-price/index.html``.  This is the full
     cross-product settlement file ("rb_e<YYYYMMDD>.csv") and it contains
     all open ``FUT_TOA3M_*`` 3-Month TONA futures rows.
  2. JPX **special-quotation HTML** (the just-expired-contract SQ table).
     A fallback when the daily file isn't reachable.
  3. ``investpy`` if installed.

ASX IB (in order):
  1. ASX **Markit research API** (the JSON endpoint that the SPA calls
     client-side).  Returns the full open strip -- typically all 18
     consecutive monthly contracts in a single response.
  2. ASX short-term-derivatives **HTML scrape** (legacy fallback for the
     rare day the page gets server-rendered with a table).
  3. ``investpy`` ``2YIB`` (AUD 30-Day Interbank Futures front-month
     close).  Synthesises a single-row strip whose expiry is anchored to
     the end of the next calendar month from the valuation date.  Enough
     to bootstrap a single-node cash-only curve.

Implementation notes
--------------------
The primary scrapes use ``urllib.request`` (project policy -- no
``requests`` dep) plus pandas' built-in ``read_html`` / ``read_csv``.

If the optional ``investpy`` package is installed, we fall back to it for
both scrapers; otherwise that fallback is a silent no-op.  ``investpy`` is
imported lazily inside a try/except so this module imports cleanly even
when it isn't present.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib import parse as urlparse
from urllib import request as urlrequest

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache + schema constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "exchange_rfr_cache"

_RFR_COLS = ["contract", "expiry_years", "price", "implied_rate"]

# 24-hour cache freshness window.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Calendar month codes used by 30-Day IB futures (one contract per month).
# F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct,
# X=Nov, Z=Dec -- the standard CME monthly code set.
_MONTH_CODE_BY_MONTH = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
_MONTH_BY_CODE = {v: k for k, v in _MONTH_CODE_BY_MONTH.items()}

# Source URLs.
_JPX_SETTLEMENT_INDEX_URL = (
    "https://www.jpx.co.jp/english/markets/derivatives/"
    "settlement-price/index.html"
)
_JPX_TONA_URL = (
    "https://www.jpx.co.jp/english/markets/derivatives/special-quotation/"
    "index.html"
)
_ASX_IB_URL = (
    "https://www2.asx.com.au/markets/trade-our-derivatives-market/"
    "derivatives-market-prices/short-term-derivatives"
)
# The ASX SPA fetches the full IB futures strip via this Markit research
# JSON endpoint.  The host + path were discovered by inspecting the SPA's
# JS bundle (see _ASX_IB_URL above).  The endpoint requires three
# (otherwise unused) chart-sizing query parameters; without them the
# server returns 400 Bad Request.
_ASX_IB_API_URL = (
    "https://asx.api.markitdigital.com/asx-research/1.0/"
    "derivatives/interest-rate/IB/futures"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 fx-project/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,text/csv,*/*",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _empty_rfr_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical RFR-strip schema."""
    return pd.DataFrame({
        c: pd.Series(dtype="object" if c == "contract" else "float64")
        for c in _RFR_COLS
    })


def _http_get(url: str, timeout: int = 15) -> str:
    """Fetch ``url`` and return the body as text.  Raises on HTTP failure."""
    req = urlrequest.Request(url, headers=_BROWSER_HEADERS)
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _third_wednesday(year: int, month: int) -> _dt.date:
    """Third Wednesday of ``year``-``month`` (TONA futures' IMM date)."""
    first = _dt.date(year, month, 1)
    # weekday(): Monday=0 ... Wednesday=2.
    offset = (2 - first.weekday()) % 7
    first_wed = first + _dt.timedelta(days=offset)
    return first_wed + _dt.timedelta(days=14)


def _month_end(year: int, month: int) -> _dt.date:
    """Last calendar day of ``year``-``month``."""
    if month == 12:
        first_next = _dt.date(year + 1, 1, 1)
    else:
        first_next = _dt.date(year, month + 1, 1)
    return first_next - _dt.timedelta(days=1)


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    """Add ``n`` months to (year, month).  Handles year rollover."""
    idx = (year * 12 + (month - 1)) + n
    return idx // 12, (idx % 12) + 1


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Return cached strip if file exists and is fresh; otherwise None."""
    if not path.exists():
        return None
    try:
        age = _dt.datetime.now().timestamp() - path.stat().st_mtime
    except OSError:
        return None
    if age > _CACHE_TTL_SECONDS:
        logger.debug("Exchange-RFR cache stale: %s", path)
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.debug("Failed reading cache %s: %s", path, exc)
        return None
    # Validate schema; if it doesn't match, ignore the cache.
    if list(df.columns) != _RFR_COLS:
        return None
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist the strip as parquet (best effort -- log + continue on error)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as exc:
        logger.debug("Failed writing cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# JPX TONA -- shared parsing helpers
# ---------------------------------------------------------------------------
# Contract symbol regex used on the JPX special-quotation page (legacy
# fixture).  Matches month name + 2/4-digit year.
_JPX_MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_JPX_CONTRACT_RE = re.compile(
    r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s\-/]*"
    r"(\d{2,4})",
    re.IGNORECASE,
)


def _parse_jpx_contract_label(label: str) -> Optional[tuple[int, int]]:
    """Extract ``(year, month)`` from a JPX contract description.

    Returns ``None`` if no month-year pattern can be found.  Two-digit
    years are rebased to the 2000s.
    """
    if not isinstance(label, str):
        return None
    m = _JPX_CONTRACT_RE.search(label)
    if m is None:
        return None
    month = _JPX_MONTH_NAMES[m.group(1).upper()]
    yraw = m.group(2)
    year = int(yraw)
    if year < 100:
        year += 2000
    return year, month


def _make_tona_row(
    year: int, month: int, price: float, val_date: _dt.date,
) -> Optional[dict]:
    """Build one strip-row dict for a TONA contract.

    ``year``/``month`` is the IMM month (third-Wednesday month).  Returns
    ``None`` if the resulting expiry is non-positive or the price is out
    of range.
    """
    if not (50.0 <= price <= 105.0):
        return None
    try:
        imm = _third_wednesday(year, month)
    except ValueError:
        return None
    expiry_years = (imm - val_date).days / 365.25
    if expiry_years <= 0:
        return None
    code = _MONTH_CODE_BY_MONTH[month]
    symbol = f"TY{code}{year % 10}"
    return {
        "contract": symbol,
        "expiry_years": float(expiry_years),
        "price": float(price),
        "implied_rate": (100.0 - float(price)) / 100.0,
    }


# ---------------------------------------------------------------------------
# TONA source 1: JPX daily-settlement CSV
# ---------------------------------------------------------------------------
# The JPX index page links to a single CSV named like
# ``rb_e20260501.csv``.  We discover the link with a regex on the index
# HTML rather than guessing a date, because non-business days are skipped.
_JPX_CSV_HREF_RE = re.compile(
    r"href=\"([^\"]+rb_e\d{8}\.csv)\"",
    re.IGNORECASE,
)

# JPX CSV row signature for 3-Month TONA: FUT_TOA3M_<YYMMDD> with
# Underlying Name "3-Month TONA".
_JPX_TONA_NAME_RE = re.compile(r"^FUT_TOA3M_(\d{6})$", re.IGNORECASE)


def _find_jpx_settlement_csv_url(html: str) -> Optional[str]:
    """Pull the absolute URL of the daily-settlement CSV out of the
    settlement-price index HTML.  Returns ``None`` if no link is found.
    """
    m = _JPX_CSV_HREF_RE.search(html)
    if m is None:
        return None
    href = m.group(1)
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.jpx.co.jp" + href
    return (
        "https://www.jpx.co.jp/english/markets/derivatives/"
        "settlement-price/" + href
    )


def _parse_jpx_settlement_csv(text: str, val_date: _dt.date) -> pd.DataFrame:
    """Parse the JPX daily-settlement CSV and pull out 3-Month TONA rows.

    The JPX file has a 2-row preamble and then a header row with columns:
        Issue Code, Issue Name, Put/Call, Contract Month, Strike Price,
        Settlement Price, Theoretical Price, Underlying Price, Volatility,
        Interest Rate, Days until Maturity, Underlying Name

    For each row whose Underlying Name is "3-Month TONA" we derive the
    IMM date from the issue-name suffix (``FUT_TOA3M_<YYMMDD>``: that
    YYMMDD is the last trading day, which is the business day before the
    3rd Wednesday of the IMM month).  We compute IMM-month from that
    date.
    """
    if not text or not text.strip():
        return _empty_rfr_df()

    try:
        # Skip the two preamble lines.  pandas can do this with skiprows=2.
        df = pd.read_csv(io.StringIO(text), skiprows=2)
    except Exception as exc:
        logger.debug("JPX CSV parse failed: %s", exc)
        return _empty_rfr_df()

    # Defensive: locate the columns we need by name.
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    name_col = None
    price_col = None
    underlying_col = None
    for key in cols_lower:
        if name_col is None and "issue name" in key:
            name_col = cols_lower[key]
        if price_col is None and "settlement price" in key:
            price_col = cols_lower[key]
        if underlying_col is None and "underlying name" in key:
            underlying_col = cols_lower[key]

    if name_col is None or price_col is None or underlying_col is None:
        logger.debug(
            "JPX CSV missing expected columns (name=%s price=%s underlying=%s)",
            name_col, price_col, underlying_col,
        )
        return _empty_rfr_df()

    rows: list[dict] = []
    for _, r in df.iterrows():
        underlying = str(r[underlying_col]).strip().upper()
        if "TONA" not in underlying or "3-MONTH" not in underlying.replace(
            " ", "-",
        ):
            continue
        issue_name = str(r[name_col]).strip()
        m = _JPX_TONA_NAME_RE.match(issue_name)
        if m is None:
            continue
        ymd = m.group(1)
        try:
            ltd = _dt.date(
                2000 + int(ymd[0:2]),
                int(ymd[2:4]),
                int(ymd[4:6]),
            )
        except ValueError:
            continue
        # Last trading day = business day before 3rd Wednesday of IMM
        # month.  IMM month = month of (ltd + a few days), since LTD is
        # at most a couple of days before the 3rd Wed.
        imm_year, imm_month = ltd.year, ltd.month
        try:
            third_wed = _third_wednesday(imm_year, imm_month)
        except ValueError:
            continue
        # Sanity check: LTD should be within the 7 days before 3rd Wed.
        if not (0 <= (third_wed - ltd).days <= 7):
            # If LTD falls outside this window, skip -- something odd.
            continue
        try:
            price = float(str(r[price_col]).replace(",", "").strip())
        except (ValueError, TypeError):
            continue
        row = _make_tona_row(imm_year, imm_month, price, val_date)
        if row is not None:
            rows.append(row)

    if not rows:
        return _empty_rfr_df()
    out = pd.DataFrame(rows, columns=_RFR_COLS)
    out = out.drop_duplicates(subset="contract", keep="first")
    return out.sort_values("expiry_years").reset_index(drop=True)


def _try_jpx_settlement_csv(val_date: _dt.date) -> pd.DataFrame:
    """Source 1 for TONA: discover + fetch the daily-settlement CSV."""
    try:
        html = _http_get(_JPX_SETTLEMENT_INDEX_URL, timeout=15)
    except Exception as exc:
        logger.info("JPX settlement-index fetch failed: %s", exc)
        return _empty_rfr_df()
    csv_url = _find_jpx_settlement_csv_url(html)
    if csv_url is None:
        logger.info("JPX settlement-index: no CSV link found")
        return _empty_rfr_df()
    try:
        text = _http_get(csv_url, timeout=20)
    except Exception as exc:
        logger.info("JPX settlement CSV fetch failed: %s", exc)
        return _empty_rfr_df()
    return _parse_jpx_settlement_csv(text, val_date)


# ---------------------------------------------------------------------------
# TONA source 2: JPX special-quotation HTML (legacy fixture)
# ---------------------------------------------------------------------------
def _parse_jpx_html_tables(html: str, val_date: _dt.date) -> pd.DataFrame:
    """Parse the JPX special-quotation HTML for TONA settlement rows.

    Used both by the live SQ-page fallback and by the existing test
    fixtures.  Scans every HTML table on the page, drops those that don't
    look like TONA tables, and pulls out (contract_label, settlement_price)
    rows.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as exc:
        logger.debug("pd.read_html failed for JPX page: %s", exc)
        return _empty_rfr_df()

    rows: list[dict] = []
    for tbl in tables:
        if tbl.empty or tbl.shape[1] < 2:
            continue
        # Look for a "TONA" mention either in the header row or the first
        # data column to confirm this is a TONA table.
        flat = " ".join(str(c) for c in tbl.columns).upper()
        first_col = tbl.iloc[:, 0].astype(str).str.upper()
        is_tona = "TONA" in flat or first_col.str.contains("TONA").any()
        if not is_tona:
            continue

        # Heuristic: contract label in the first column, settlement price
        # in a column whose header contains "PRICE" or "SETTLE".  Fall
        # back to the last numeric column.
        price_col = None
        for c in tbl.columns:
            cu = str(c).upper()
            if "SETTLE" in cu or "PRICE" in cu or "SQ" in cu:
                price_col = c
                break
        if price_col is None:
            # Last column is usually the settlement value.
            price_col = tbl.columns[-1]

        for _, r in tbl.iterrows():
            label = str(r.iloc[0])
            ym = _parse_jpx_contract_label(label)
            if ym is None:
                continue
            year, month = ym
            try:
                price = float(str(r[price_col]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            row = _make_tona_row(year, month, price, val_date)
            if row is not None:
                rows.append(row)

    if not rows:
        return _empty_rfr_df()

    out = pd.DataFrame(rows, columns=_RFR_COLS)
    out = out.drop_duplicates(subset="contract", keep="first")
    return out.sort_values("expiry_years").reset_index(drop=True)


def _try_jpx_sq_html(val_date: _dt.date) -> pd.DataFrame:
    """Source 2 for TONA: scrape the special-quotation HTML page."""
    try:
        html = _http_get(_JPX_TONA_URL, timeout=15)
    except Exception as exc:
        logger.info("JPX special-quotation HTML fetch failed: %s", exc)
        return _empty_rfr_df()
    return _parse_jpx_html_tables(html, val_date)


# ---------------------------------------------------------------------------
# TONA source 3: investpy
# ---------------------------------------------------------------------------
def _try_investpy_tona(val_date: _dt.date) -> pd.DataFrame:
    """Optional fallback via investpy.  Returns empty DF if unavailable
    or if no TONA series surfaces.

    investpy (Investing.com mirror) historically does not carry TONA
    futures, but versions vary -- we search lazily and synthesize a
    single-row strip if we get a usable quote.
    """
    try:
        import investpy  # type: ignore
    except Exception:
        return _empty_rfr_df()

    candidates: list = []
    for q in ("3 Month TONA", "TONA Futures", "Tokyo Overnight Average"):
        try:
            res = investpy.search_quotes(  # type: ignore[attr-defined]
                text=q, n_results=5,
            )
        except Exception:
            continue
        if not res:
            continue
        for r in res:
            name = (getattr(r, "name", "") or "").upper()
            if "TONA" in name:
                candidates.append(r)
        if candidates:
            break

    for cand in candidates:
        try:
            data = cand.retrieve_recent_data()
        except Exception:
            continue
        if not isinstance(data, pd.DataFrame) or data.empty:
            continue
        try:
            close = float(data["Close"].iloc[-1])
        except Exception:
            continue
        # Synthesise a single front-month strip row at the next IMM.
        nxt_year, nxt_month = _add_months(val_date.year, val_date.month, 1)
        # Round to the nearest quarterly month (March, Jun, Sep, Dec).
        while nxt_month not in (3, 6, 9, 12):
            nxt_year, nxt_month = _add_months(nxt_year, nxt_month, 1)
        row = _make_tona_row(nxt_year, nxt_month, close, val_date)
        if row is None:
            continue
        return pd.DataFrame([row], columns=_RFR_COLS)

    return _empty_rfr_df()


# ---------------------------------------------------------------------------
# fetch_tona_strip
# ---------------------------------------------------------------------------
def fetch_tona_strip(date_str: str) -> pd.DataFrame:
    """Fetch OSE 3M TONA futures (JPX) settlement strip.

    Tries multiple sources in order: (1) the JPX daily-settlement CSV,
    (2) the JPX special-quotation HTML page, (3) ``investpy`` if
    installed.  Returns the first non-empty result, or an empty
    DataFrame with the canonical schema if none of them yields data.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).

    Returns
    -------
    pd.DataFrame
        Columns ``['contract', 'expiry_years', 'price', 'implied_rate']``
        sorted ascending in expiry.  Never raises.
    """
    try:
        val_date = _dt.date.fromisoformat(date_str)
    except (ValueError, TypeError):
        logger.warning("fetch_tona_strip: invalid date_str %r", date_str)
        return _empty_rfr_df()

    cache_path = CACHE_DIR / f"tona_ty_strip_{date_str}.parquet"
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    sources = [
        ("jpx_csv", _try_jpx_settlement_csv),
        ("jpx_sq_html", _try_jpx_sq_html),
        ("investpy", _try_investpy_tona),
    ]
    out = _empty_rfr_df()
    for name, fn in sources:
        try:
            candidate = fn(val_date)
        except Exception as exc:
            logger.info("TONA source %s raised: %s", name, exc)
            continue
        if isinstance(candidate, pd.DataFrame) and len(candidate) > 0:
            logger.info("TONA strip provided by source %s (%d rows)",
                        name, len(candidate))
            out = candidate
            break
        logger.info("TONA source %s returned 0 rows", name)

    if not out.empty:
        _write_cache(out, cache_path)
    return out


# ---------------------------------------------------------------------------
# ASX IB -- shared parsing helpers
# ---------------------------------------------------------------------------
# ASX month codes for IB futures: published as e.g. "IBM6" (Jun 2026),
# "IBN6" (Jul 2026), etc.  The 30-Day IB future references the average
# overnight cash rate over the calendar month of expiry, so we anchor
# expiry_years to month-end.
_ASX_CONTRACT_RE = re.compile(
    r"\bIB([FGHJKMNQUVXZ])(\d)\b",
    re.IGNORECASE,
)
# Long-form descriptions: "Jun 26", "June 2026", "Jun-26".
_ASX_MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "JUNE": 6, "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9,
    "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}
_ASX_LONG_RE = re.compile(
    r"(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|"
    r"OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|"
    r"SEP|OCT|NOV|DEC)[\s\-/]*(\d{2,4})",
    re.IGNORECASE,
)


def _parse_asx_contract_label(
    label: str, val_date: _dt.date,
) -> Optional[tuple[str, int, int]]:
    """Extract ``(symbol, year, month)`` from an ASX IB contract label."""
    if not isinstance(label, str):
        return None

    m = _ASX_CONTRACT_RE.search(label)
    if m is not None:
        code = m.group(1).upper()
        if code not in _MONTH_BY_CODE:
            return None
        month = _MONTH_BY_CODE[code]
        decade_base = (val_date.year // 10) * 10
        year = decade_base + int(m.group(2))
        if year < val_date.year - 5:
            year += 10
        symbol = f"IB{code}{year % 10}"
        return symbol, year, month

    m = _ASX_LONG_RE.search(label)
    if m is not None:
        month = _ASX_MONTH_NAMES[m.group(1).upper()]
        yraw = m.group(2)
        year = int(yraw)
        if year < 100:
            year += 2000
        code = _MONTH_CODE_BY_MONTH[month]
        symbol = f"IB{code}{year % 10}"
        return symbol, year, month

    return None


def _make_ib_row(
    symbol: str, year: int, month: int, price: float, val_date: _dt.date,
) -> Optional[dict]:
    """Build one strip-row dict for an ASX IB contract."""
    if not (90.0 <= price <= 105.0):
        return None
    expiry = _month_end(year, month)
    expiry_years = (expiry - val_date).days / 365.25
    if expiry_years <= 0:
        return None
    return {
        "contract": symbol,
        "expiry_years": float(expiry_years),
        "price": float(price),
        "implied_rate": (100.0 - float(price)) / 100.0,
    }


# ---------------------------------------------------------------------------
# ASX source 1: Markit research JSON API
# ---------------------------------------------------------------------------
# Each row in ``data.items`` looks like:
#   {
#     "dateExpiry": "2026-05-27",
#     "priceLastTrade": 95.69,
#     "pricePreviousSettlement": 95.695,
#     "symbol": "IBK2026",
#     ...
#   }
# We prefer ``pricePreviousSettlement`` (the published EOD settlement)
# and fall back to ``priceLastTrade`` if it's missing.
_ASX_API_SYMBOL_RE = re.compile(
    r"^IB([FGHJKMNQUVXZ])(\d{2,4})$",
    re.IGNORECASE,
)


def _parse_asx_api_symbol(
    sym: str, expiry_iso: Optional[str], val_date: _dt.date,
) -> Optional[tuple[str, int, int]]:
    """Resolve ``(short_symbol, year, month)`` from a Markit feed entry.

    ``sym`` is the long-form ASX symbol (``"IBK2026"``).  We prefer the
    explicit ``expiry_iso`` if supplied (it disambiguates the year), and
    fall back to parsing the long-form symbol.
    """
    year = month = None
    if isinstance(expiry_iso, str):
        try:
            d = _dt.date.fromisoformat(expiry_iso[:10])
            year, month = d.year, d.month
        except ValueError:
            year = month = None
    if year is None or month is None:
        m = _ASX_API_SYMBOL_RE.match(sym or "")
        if m is None:
            return None
        code = m.group(1).upper()
        if code not in _MONTH_BY_CODE:
            return None
        month = _MONTH_BY_CODE[code]
        yraw = int(m.group(2))
        year = yraw if yraw >= 1000 else 2000 + yraw

    code = _MONTH_CODE_BY_MONTH[month]
    short_symbol = f"IB{code}{year % 10}"
    return short_symbol, year, month


def _parse_asx_api_payload(
    payload: dict, val_date: _dt.date,
) -> pd.DataFrame:
    """Pull the IB strip out of the Markit ``/futures`` JSON payload."""
    if not isinstance(payload, dict):
        return _empty_rfr_df()
    data = payload.get("data") or {}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        return _empty_rfr_df()

    rows: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = it.get("symbol") or ""
        expiry_iso = it.get("dateExpiry")
        parsed = _parse_asx_api_symbol(sym, expiry_iso, val_date)
        if parsed is None:
            continue
        short_symbol, year, month = parsed

        # Settlement preferred; last-trade is the live fallback.
        price = it.get("pricePreviousSettlement")
        if price is None or (isinstance(price, float) and price != price):
            price = it.get("priceLastTrade")
        if price is None:
            continue
        try:
            price_f = float(price)
        except (ValueError, TypeError):
            continue
        row = _make_ib_row(short_symbol, year, month, price_f, val_date)
        if row is not None:
            rows.append(row)

    if not rows:
        return _empty_rfr_df()

    out = pd.DataFrame(rows, columns=_RFR_COLS)
    out = out.drop_duplicates(subset="contract", keep="first")
    return out.sort_values("expiry_years").reset_index(drop=True)


def _try_asx_api(val_date: _dt.date) -> pd.DataFrame:
    """Source 1 for ASX IB: the Markit research JSON API."""
    qs = urlparse.urlencode({"days": 5, "height": 600, "width": 800})
    url = f"{_ASX_IB_API_URL}?{qs}"
    try:
        text = _http_get(url, timeout=15)
    except Exception as exc:
        logger.info("ASX IB API fetch failed: %s", exc)
        return _empty_rfr_df()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.info("ASX IB API JSON parse failed: %s", exc)
        return _empty_rfr_df()
    return _parse_asx_api_payload(payload, val_date)


# ---------------------------------------------------------------------------
# ASX source 2: short-term-derivatives HTML scrape (legacy)
# ---------------------------------------------------------------------------
def _parse_asx_html_tables(html: str, val_date: _dt.date) -> pd.DataFrame:
    """Parse the ASX short-term-derivatives HTML for IB settlement rows."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as exc:
        logger.debug("pd.read_html failed for ASX page: %s", exc)
        return _empty_rfr_df()

    rows: list[dict] = []
    for tbl in tables:
        if tbl.empty or tbl.shape[1] < 2:
            continue
        flat = " ".join(str(c) for c in tbl.columns).upper()
        first_col = tbl.iloc[:, 0].astype(str).str.upper()
        is_ib = (
            "IB" in flat
            or "INTERBANK" in flat
            or first_col.str.contains("IB").any()
            or first_col.str.contains("INTERBANK").any()
        )
        if not is_ib:
            continue

        price_col = None
        for c in tbl.columns:
            cu = str(c).upper()
            if "SETTLE" in cu or "LAST" in cu or "PRICE" in cu:
                price_col = c
                break
        if price_col is None:
            price_col = tbl.columns[-1]

        for _, r in tbl.iterrows():
            label = str(r.iloc[0])
            parsed = _parse_asx_contract_label(label, val_date)
            if parsed is None:
                continue
            symbol, year, month = parsed
            try:
                price = float(str(r[price_col]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            row = _make_ib_row(symbol, year, month, price, val_date)
            if row is not None:
                rows.append(row)

    if not rows:
        return _empty_rfr_df()

    out = pd.DataFrame(rows, columns=_RFR_COLS)
    out = out.drop_duplicates(subset="contract", keep="first")
    return out.sort_values("expiry_years").reset_index(drop=True)


def _try_asx_html(val_date: _dt.date) -> pd.DataFrame:
    """Source 1 for ASX IB: scrape the short-term-derivatives HTML page."""
    try:
        html = _http_get(_ASX_IB_URL, timeout=15)
    except Exception as exc:
        logger.info("ASX IB HTML fetch failed: %s", exc)
        return _empty_rfr_df()
    return _parse_asx_html_tables(html, val_date)


# ---------------------------------------------------------------------------
# ASX source 2: investpy 2YIB (AUD 30-Day Interbank Futures front-month)
# ---------------------------------------------------------------------------
def _try_investpy_aonia(val_date: _dt.date) -> pd.DataFrame:
    """Fallback via investpy.

    Looks up the ``AUD 30 day Interbank Futures`` quote (symbol
    ``2YIB``) and synthesises a single-row front-month strip whose
    expiry is anchored to the next calendar month-end after
    ``val_date`` (the natural 30-day IB averaging window).

    Returns an empty DataFrame if investpy is not installed, the
    search fails, or the returned series is empty.
    """
    try:
        import investpy  # type: ignore
    except Exception:
        return _empty_rfr_df()

    target = None
    try:
        res = investpy.search_quotes(  # type: ignore[attr-defined]
            text="30 Day Interbank", n_results=10,
        )
    except Exception as exc:
        logger.info("investpy.search_quotes failed: %s", exc)
        return _empty_rfr_df()
    if not res:
        return _empty_rfr_df()
    for r in res:
        sym = (getattr(r, "symbol", "") or "").upper()
        country = (getattr(r, "country", "") or "").lower()
        if sym == "2YIB" or (country == "australia" and "INTERBANK" in
                             (getattr(r, "name", "") or "").upper()):
            target = r
            break
    if target is None:
        return _empty_rfr_df()

    try:
        data = target.retrieve_recent_data()
    except Exception as exc:
        logger.info("investpy retrieve_recent_data failed: %s", exc)
        return _empty_rfr_df()
    if not isinstance(data, pd.DataFrame) or data.empty:
        return _empty_rfr_df()

    # Find the close price closest to (but not after) val_date.
    try:
        idx = pd.to_datetime(data.index).date
        mask = idx <= val_date
        if mask.any():
            close = float(data.loc[mask].iloc[-1]["Close"])
        else:
            close = float(data["Close"].iloc[-1])
    except Exception as exc:
        logger.info("investpy 2YIB price extraction failed: %s", exc)
        return _empty_rfr_df()

    # Anchor expiry to the next calendar month-end (the 30-day IB
    # averages over a single calendar month).
    nxt_year, nxt_month = _add_months(val_date.year, val_date.month, 1)
    code = _MONTH_CODE_BY_MONTH[nxt_month]
    symbol = f"IB{code}{nxt_year % 10}"
    row = _make_ib_row(symbol, nxt_year, nxt_month, close, val_date)
    if row is None:
        return _empty_rfr_df()
    return pd.DataFrame([row], columns=_RFR_COLS)


# ---------------------------------------------------------------------------
# fetch_aonia_ib_strip
# ---------------------------------------------------------------------------
def fetch_aonia_ib_strip(date_str: str) -> pd.DataFrame:
    """Fetch ASX 30-Day Interbank Cash Rate Futures (IB) settlement strip.

    Tries multiple sources in order: (1) the ASX Markit research JSON
    API (the same endpoint the SPA hits client-side -- typically returns
    all 18 consecutive monthly contracts), (2) the ASX short-term-
    derivatives HTML page (legacy fallback), (3) ``investpy`` ``2YIB``
    (AUD 30-Day Interbank Futures front-month).  Returns the first
    non-empty result, or an empty DataFrame with the canonical schema
    if none yields data.

    Parameters
    ----------
    date_str : str
        Valuation date in ISO format (``"YYYY-MM-DD"``).

    Returns
    -------
    pd.DataFrame
        Columns ``['contract', 'expiry_years', 'price', 'implied_rate']``
        sorted ascending in expiry.  Never raises.
    """
    try:
        val_date = _dt.date.fromisoformat(date_str)
    except (ValueError, TypeError):
        logger.warning("fetch_aonia_ib_strip: invalid date_str %r", date_str)
        return _empty_rfr_df()

    cache_path = CACHE_DIR / f"aonia_ib_strip_{date_str}.parquet"
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    sources = [
        ("asx_api", _try_asx_api),
        ("asx_html", _try_asx_html),
        ("investpy", _try_investpy_aonia),
    ]
    out = _empty_rfr_df()
    for name, fn in sources:
        try:
            candidate = fn(val_date)
        except Exception as exc:
            logger.info("ASX IB source %s raised: %s", name, exc)
            continue
        if isinstance(candidate, pd.DataFrame) and len(candidate) > 0:
            logger.info("ASX IB strip provided by source %s (%d rows)",
                        name, len(candidate))
            out = candidate
            break
        logger.info("ASX IB source %s returned 0 rows", name)

    if not out.empty:
        _write_cache(out, cache_path)
    return out
