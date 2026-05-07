"""
Foreign OIS Curve Loader
========================

Fetches real foreign-currency OIS / sovereign zero curves from free public APIs:

    EUR -- ECB legacy SDMX (Tier 1) with FRED euro-area proxy fallback (Tier 2)
    GBP -- BoE Interactive Statistical Database (Tier 1) with FRED proxy (Tier 2)
    JPY -- FRED Japan term structure (TONA-equiv overnight + 3M interbank + 10Y govt)
    AUD -- RBA F1 statistics CSV (Tier 1) with FRED long-end fillers (Tier 2)

For each non-JPY currency the loader tries a primary source; if it returns
no tenors, it falls back to FRED proxies.  If everything fails, an empty
dict is returned and the caller falls back to a flat-rate proxy.

Public API
----------
fetch_foreign_ois_curve(currency, date_str) -> dict[float, float]
    Returns {tenor_years: zero_rate_decimal}.  Empty dict on total failure.
    Caches the result as parquet under data/foreign_ois_cache/.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional
from urllib import request as urlrequest

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------
CACHE_DIR = config.DATA_DIR / "foreign_ois_cache"

SUPPORTED_CURRENCIES = ("EUR", "GBP", "JPY", "AUD")

# ECB SDW series codes (one per tenor) and their tenor in years
ECB_SERIES = {
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M": 0.25,
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_6M": 0.5,
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y": 1.0,
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y": 2.0,
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y": 5.0,
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y": 10.0,
}

# BoE IADB series codes -> tenor (years)
# IUDSOIA = SONIA fixing (overnight)
# IUDMNZC = nominal spot 1Y
# IUDMNPC = nominal spot 5Y
# IUDMNHC = nominal spot 10Y
BOE_SERIES_MAP = {
    "IUDSOIA": 1.0 / 365.0,
    "IUDMNZC": 1.0,
    "IUDMNPC": 5.0,
    "IUDMNHC": 10.0,
}

# RBA F1 column codes -> tenor (years).
# RBA changes the publication suffix occasionally (e.g. trailing "D" for
# daily series).  We match by *prefix* in _parse_rba_csv so both the legacy
# and current Series IDs work.
RBA_SERIES_MAP = {
    "FIRMMCRT": 1.0 / 365.0,    # cash rate (current ID is FIRMMCRTD)
    "FIRMMBAB90": 0.25,         # 90d bank bill (current ID is FIRMMBAB90D)
    "FCMYGBAG3": 3.0,           # 3y govt (also appears in F2)
    "FCMYGBAG5": 5.0,           # 5y govt (also appears in F2)
    "FCMYGBAG10": 10.0,         # 10y govt (also appears in F2)
}

# RBA F1 OIS column codes -> tenor (years).  These are looked up separately
# from the main F1 short-end so that a populated 6M OIS quote can supersede
# the 3M BAB anchor when both are present.  Recent F1 publications sometimes
# omit the OIS columns entirely; the helper degrades gracefully.
RBA_F1_OIS_MAP = {
    "FIRMMOIS1": 1.0 / 12.0,    # 1m OIS  (FIRMMOIS1D)
    "FIRMMOIS3": 0.25,          # 3m OIS  (FIRMMOIS3D)
    "FIRMMOIS6": 0.5,           # 6m OIS  (FIRMMOIS6D)
}

# RBA F2 column codes -> tenor (years).  These are the Australian government
# bond yield series; F2 does not publish a 1Y CGS, so the OIS-CGS spread is
# anchored by comparing 90D BAB (a near-OIS rate) to a 1Y CGS implied by
# linear interp between the cash rate and 2Y CGS.
RBA_F2_SERIES_MAP = {
    "FCMYGBAG2": 2.0,           # 2y govt (FCMYGBAG2D)
    "FCMYGBAG3": 3.0,           # 3y govt (FCMYGBAG3D)
    "FCMYGBAG5": 5.0,           # 5y govt (FCMYGBAG5D)
    "FCMYGBAG10": 10.0,         # 10y govt (FCMYGBAG10D)
}

# Months for BoE URL
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Common HTTP headers for fronted endpoints (BoE, RBA, ECB legacy).  A
# realistic Mozilla User-Agent and an Accept header tend to bypass naive
# WAF rules that reject the bare "Python-urllib" UA.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 fx-project/1.0"
    ),
    "Accept": "text/csv;version=1.0.0, text/csv, */*",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _http_get(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 15,
) -> str:
    """Thin urllib wrapper -- returns response body as text.

    Parameters
    ----------
    url : str
    headers : dict | None
        Optional HTTP headers (e.g. browser User-Agent + Accept).
    timeout : int
    """
    final_headers = {"User-Agent": "fx-project/1.0"}
    if headers:
        final_headers.update(headers)
    req = urlrequest.Request(url, headers=final_headers)
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# ECB (EUR) -- SDW yield curve CSV
# ---------------------------------------------------------------------------
def _parse_ecb_csv(csv_text: str) -> dict[float, float]:
    """Parse an ECB SDW CSV -- one or more series rows.

    Returns a dict {0.0: rate}; the calling site overwrites the tenor key.
    """
    out: dict[float, float] = {}
    if not csv_text or not csv_text.strip():
        return out
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        logger.debug("ECB CSV parse failed: %s", exc)
        return out

    if "OBS_VALUE" not in df.columns:
        return out

    series = pd.to_numeric(df["OBS_VALUE"], errors="coerce").dropna()
    if series.empty:
        return out
    last_val = float(series.iloc[-1]) / 100.0
    out[0.0] = last_val
    return out


def _ecb_sdw_eur_ois(date_str: str) -> dict[float, float]:
    """Tier 1: ECB legacy SDMX endpoint (sdw-wsrest.ecb.europa.eu).

    The newer data-api.ecb.europa.eu host is blocked by a WAF for non-
    browser clients, so we use the legacy host with a Mozilla UA.
    """
    base = "https://sdw-wsrest.ecb.europa.eu/service/data"
    curve: dict[float, float] = {}
    for series_code, tenor in ECB_SERIES.items():
        # Series codes are dotted; split into dataflow + key
        parts = series_code.split(".", 1)
        if len(parts) != 2:
            continue
        dataflow, key = parts[0], parts[1]
        url = (
            f"{base}/{dataflow}/{key}"
            f"?startPeriod={date_str}&endPeriod={date_str}&format=csvdata"
        )
        try:
            csv_text = _http_get(url, headers=_BROWSER_HEADERS)
        except Exception as exc:
            logger.debug("ECB legacy fetch %s failed: %s", series_code, exc)
            continue
        parsed = _parse_ecb_csv(csv_text)
        if 0.0 in parsed:
            curve[tenor] = parsed[0.0]
    return curve


def _fred_eur_proxy(date_str: str) -> dict[float, float]:
    """Tier 2 EUR: build a 3-point curve from FRED euro-area series.

    Tenors:
        overnight (1/365) -> ECBESTRVOLWGTTRMDMNRT (euro short-term rate / €STR,
                              the actual ECB reference rate, daily).  Falls back
                              to ECBDFR (deposit facility rate, daily) for dates
                              where €STR is unavailable -- DFR sits ~7-10 bp
                              below €STR so this is a small but real bias.
        0.25 (3M)         -> IR3TIB01EZM156N (euro-area 3M interbank, monthly)
        10.0              -> IRLTLT01EZM156N (euro-area 10Y govt, monthly)
    """
    api_key = config.get_fred_api_key()
    if not api_key:
        return {}

    # Each entry: (tenor, [primary_series_id, fallback_series_id, ...])
    plan: list[tuple[float, list[str]]] = [
        (1.0 / 365.0, ["ECBESTRVOLWGTTRMDMNRT", "ECBDFR"]),
        (0.25, ["IR3TIB01EZM156N"]),
        (10.0, ["IRLTLT01EZM156N"]),
    ]
    curve: dict[float, float] = {}
    for tenor, series_ids in plan:
        for series_id in series_ids:
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&file_type=json&sort_order=desc&limit=5"
                f"&observation_end={date_str}"
                f"&api_key={api_key}"
            )
            try:
                json_text = _http_get(url)
            except Exception as exc:
                logger.debug("FRED EUR fetch %s failed: %s", series_id, exc)
                continue
            rate = _parse_fred_json(json_text)
            if rate is None:
                continue
            # ECBDFR has occasional zero values; skip if exactly 0.
            if series_id == "ECBDFR" and rate == 0.0:
                continue
            curve[tenor] = rate
            break  # primary succeeded, skip fallbacks for this tenor
    return curve


# ---------------------------------------------------------------------------
# BoE (GBP) -- IADB CSV
# ---------------------------------------------------------------------------
def _parse_boe_csv(csv_text: str) -> dict[float, float]:
    """Parse a BoE IADB CSV containing SONIA + spot curve series."""
    out: dict[float, float] = {}
    if not csv_text or not csv_text.strip():
        return out
    # Reject HTML error pages
    head = csv_text.lstrip()[:200].lower()
    if head.startswith("<!doctype") or head.startswith("<html"):
        return out
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        logger.debug("BoE CSV parse failed: %s", exc)
        return out

    series_cols = [c for c in BOE_SERIES_MAP if c in df.columns]
    if not series_cols:
        return out

    for col in series_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_clean = df.dropna(subset=series_cols, how="all")
    if df_clean.empty:
        return out

    last_row = df_clean.iloc[-1]
    for col in series_cols:
        val = last_row[col]
        if pd.notna(val):
            tenor = BOE_SERIES_MAP[col]
            out[tenor] = float(val) / 100.0
    return out


def _boe_iadb_gbp_ois(date_str: str) -> dict[float, float]:
    """Tier 1: BoE IADB CSV with browser headers."""
    try:
        year = int(date_str[0:4])
        month = int(date_str[5:7])
        day = int(date_str[8:10])
    except (ValueError, IndexError):
        return {}

    mon = _MONTHS[month - 1]
    series_codes = ",".join(BOE_SERIES_MAP.keys())
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
        f"?Travel=NIxAZxSUx&FromSeries=1&ToSeries=50&DAT=ALL"
        f"&FD=1&FM=Jan&FY=2020"
        f"&TD={day}&TM={mon}&TY={year}"
        f"&FNY=Y&CSVF=TT&html.x=66&html.y=26"
        f"&SeriesCodes={series_codes}&UsingCodes=Y"
    )
    try:
        csv_text = _http_get(url, headers=_BROWSER_HEADERS)
    except Exception as exc:
        logger.debug("BoE fetch failed: %s", exc)
        return {}
    return _parse_boe_csv(csv_text)


def _fred_gbp_proxy(date_str: str) -> dict[float, float]:
    """Tier 2 GBP: build a 3-point curve from FRED UK series.

    Tenors:
        overnight (1/365) -> IUDSOIA (SONIA, daily)
        0.25 (3M)         -> IR3TIB01GBM156N (UK 3M interbank, monthly)
        10.0              -> IRLTLT01GBM156N (UK 10Y govt, monthly)
    """
    api_key = config.get_fred_api_key()
    if not api_key:
        return {}

    plan = [
        ("IUDSOIA", 1.0 / 365.0),
        ("IR3TIB01GBM156N", 0.25),
        ("IRLTLT01GBM156N", 10.0),
    ]
    curve: dict[float, float] = {}
    for series_id, tenor in plan:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&file_type=json&sort_order=desc&limit=5"
            f"&observation_end={date_str}"
            f"&api_key={api_key}"
        )
        try:
            json_text = _http_get(url)
        except Exception as exc:
            logger.debug("FRED GBP fetch %s failed: %s", series_id, exc)
            continue
        rate = _parse_fred_json(json_text)
        if rate is None:
            continue
        curve[tenor] = rate
    return curve


# ---------------------------------------------------------------------------
# BoJ (JPY) -- TONA via FRED, slope synthesised
# ---------------------------------------------------------------------------
def _parse_fred_json(json_text: str) -> Optional[float]:
    """Parse FRED JSON -- return the first usable observation as decimal."""
    if not json_text or not json_text.strip():
        return None
    try:
        payload = json.loads(json_text)
    except Exception as exc:
        logger.debug("FRED JSON parse failed: %s", exc)
        return None
    obs = payload.get("observations", [])
    if not obs:
        return None
    for entry in obs:
        val_str = entry.get("value", "")
        if val_str in (".", "", None):
            continue
        try:
            return float(val_str) / 100.0
        except (TypeError, ValueError):
            continue
    return None


def _fred_lookup(series_id: str, date_str: str) -> Optional[float]:
    """Fetch the most recent FRED observation on or before date_str.

    Returns the rate as a decimal (e.g. 0.0125 for 1.25%) or None on failure.
    Shared helper used by JPY and AUD term-structure builders.
    """
    api_key = config.get_fred_api_key()
    if not api_key:
        return None
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&file_type=json&sort_order=desc&limit=1"
        f"&observation_start=2020-01-01&observation_end={date_str}"
        f"&api_key={api_key}"
    )
    try:
        body = _http_get(url)
    except Exception as exc:
        logger.debug("FRED lookup %s failed: %s", series_id, exc)
        return None
    return _parse_fred_json(body)


def _boj_tona_jpy_ois(date_str: str) -> dict[float, float]:
    """JPY term structure: BoJ policy overnight + 3M interbank + 10Y govt via FRED.

    Overnight anchor primary:  INTDSRJPM193N  (BoJ policy rate, daily-equivalent).
    Overnight anchor fallback: IRSTCI01JPM156N (JPY interbank call, monthly --
                               lags BoJ policy by ~1-2 weeks).
    """
    # (tenor, [primary_series_id, fallback_series_id, ...])
    plan: list[tuple[float, list[str]]] = [
        (1.0 / 365.0, ["INTDSRJPM193N", "IRSTCI01JPM156N"]),
        (0.25, ["IR3TIB01JPM156N"]),
        (10.0, ["IRLTLT01JPM156N"]),
    ]
    out: dict[float, float] = {}
    for tenor, series_ids in plan:
        for series_id in series_ids:
            try:
                rate = _fred_lookup(series_id, date_str)
            except Exception as exc:
                logger.warning("FRED %s failed: %s", series_id, exc)
                continue
            if rate is not None:
                out[tenor] = rate
                break  # primary succeeded, skip fallbacks for this tenor
    return out


# ---------------------------------------------------------------------------
# RBA (AUD) -- F1 statistics CSV
# ---------------------------------------------------------------------------
def _rba_match_columns(
    columns: list[str],
    series_map: Optional[dict] = None,
) -> dict[str, float]:
    """Map actual CSV column names to tenor years using prefix matching.

    RBA changes the publication suffix occasionally (e.g. trailing "D"
    for daily).  By matching on the prefix from `series_map` we tolerate
    both `FIRMMCRT` and `FIRMMCRTD`.

    `series_map` defaults to the F1 short-end map (RBA_SERIES_MAP) for
    backwards compatibility; pass RBA_F2_SERIES_MAP or RBA_F1_OIS_MAP to
    target the long-end (F2 govt yields) or the F1 OIS columns.
    """
    if series_map is None:
        series_map = RBA_SERIES_MAP
    out: dict[str, float] = {}
    for col in columns:
        if not isinstance(col, str):
            continue
        col_clean = col.strip()
        # Find the longest matching prefix to avoid e.g. matching
        # FIRMMCRT to FIRMMCRTRI (cash rate vs total return index).
        best_prefix = ""
        best_tenor: Optional[float] = None
        for prefix, tenor in series_map.items():
            if col_clean.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_tenor = tenor
        if best_tenor is not None:
            # Reject suspiciously-long suffixes (e.g. FIRMMCRTRI -> total
            # return index) by requiring the suffix to be at most 1 char
            # for the cash-rate prefix.  More conservative: only allow
            # suffix that is "" or a single uppercase letter "D".
            suffix = col_clean[len(best_prefix):]
            if suffix == "" or suffix == "D":
                out[col_clean] = best_tenor
    return out


def _parse_rba_csv(
    csv_text: str,
    date_str: str,
    series_map: Optional[dict] = None,
) -> dict[float, float]:
    """Parse an RBA F1 / F2 CSV.

    The CSV has a header block (~10 lines of metadata) before the actual
    "Series ID" header row.  We find that row, then use the Series IDs
    as column names for the data block, and map them to tenor by prefix.

    Picks the most recent row with date <= date_str.

    `series_map` selects which Series IDs to extract.  Defaults to the F1
    short-end map (RBA_SERIES_MAP); pass RBA_F2_SERIES_MAP for F2 govt
    yields or RBA_F1_OIS_MAP for the F1 OIS columns.
    """
    out: dict[float, float] = {}
    if not csv_text or not csv_text.strip():
        return out

    lines = csv_text.splitlines()
    series_id_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Series ID,") or ",Series ID," in line:
            series_id_idx = i
            break

    if series_id_idx is None:
        # Fallback: try to read the whole CSV with default header
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            return out
    else:
        # Build a fresh CSV: Series ID row as header, then all data rows
        # below (skipping any blank lines).
        header_line = lines[series_id_idx]
        # Replace the leading "Series ID" cell with a "date" name.
        first_comma = header_line.find(",")
        if first_comma == -1:
            return out
        new_header = "date" + header_line[first_comma:]

        data_lines: list[str] = []
        for raw in lines[series_id_idx + 1:]:
            if raw.strip() == "":
                continue
            data_lines.append(raw)

        if not data_lines:
            return out

        rebuilt = "\n".join([new_header] + data_lines)
        try:
            df = pd.read_csv(io.StringIO(rebuilt))
        except Exception as exc:
            logger.debug("RBA CSV parse failed: %s", exc)
            return out

    if df.empty:
        return out

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col])

    target_date = pd.to_datetime(date_str)
    df = df[df[date_col] <= target_date]
    if df.empty:
        return out

    # Map actual columns to tenors (handles trailing "D" suffix)
    column_to_tenor = _rba_match_columns(list(df.columns), series_map=series_map)
    if not column_to_tenor:
        return out

    last_row = df.sort_values(date_col).iloc[-1]
    for col, tenor in column_to_tenor.items():
        val = pd.to_numeric(last_row[col], errors="coerce")
        if pd.notna(val):
            out[tenor] = float(val) / 100.0
    return out


def _rba_f2_govt_yields(date_str: str) -> dict[float, float]:
    """Fetch RBA F2 (Capital Market Yields) Australian govt bond yields.

    Returns ``{tenor_years: yield_decimal}`` for whichever of the 2Y / 3Y /
    5Y / 10Y CGS series is populated on or before ``date_str``.  Falls back
    to ``{}`` on any fetch / parse failure -- the caller is expected to
    degrade gracefully (e.g. by skipping the F2-derived bridge points and
    using FRED long-end fillers instead).
    """
    url = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"
    try:
        csv_text = _http_get(url, headers=_BROWSER_HEADERS)
    except Exception as exc:
        logger.warning("RBA F2 fetch failed: %s", exc)
        return {}
    try:
        return _parse_rba_csv(csv_text, date_str, series_map=RBA_F2_SERIES_MAP)
    except Exception as exc:
        logger.warning("RBA F2 parse failed: %s", exc)
        return {}


def _rba_f1_ois_quotes(date_str: str, csv_text: Optional[str] = None) -> dict[float, float]:
    """Pull the F1 OIS columns (1M / 3M / 6M OIS) when published.

    Recent F1 CSVs sometimes omit the OIS columns entirely from new rows,
    so the result is often empty.  Returned as ``{tenor_years: rate_decimal}``.
    Pass ``csv_text`` to reuse a body already fetched by the caller.
    """
    if csv_text is None:
        url = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
        try:
            csv_text = _http_get(url, headers=_BROWSER_HEADERS)
        except Exception as exc:
            logger.debug("RBA F1 fetch (OIS) failed: %s", exc)
            return {}
    try:
        return _parse_rba_csv(csv_text, date_str, series_map=RBA_F1_OIS_MAP)
    except Exception as exc:
        logger.debug("RBA F1 OIS parse failed: %s", exc)
        return {}


def _aud_ois_cgs_spread(date_str: str) -> Optional[float]:
    """Backout the OIS-CGS basis at a comparable short tenor.

    Approach
    --------
    1. Pull the F1 short end -- cash rate and 90D BAB.  The 90D BAB is a
       very close approximation of the 90D OIS rate (BAB-OIS basis is
       usually <5 bp).
    2. If F1 publishes a 1Y OIS column, use it directly as the OIS anchor;
       otherwise the 90D BAB is the OIS anchor.
    3. Pull F2 to obtain the 2Y CGS yield.  F2 does not publish a 1Y CGS,
       so the 1Y CGS is approximated by linearly interpolating between the
       cash rate (treated as the very-front-end govt yield) and 2Y CGS.
    4. Spread = OIS_anchor - CGS_at_same_tenor.

    Returns the spread as a decimal (e.g. -0.0030 for -30 bp).  Falls back
    to ``None`` if F1 or F2 are unavailable, or if the inputs are insane
    (|spread| >= 200 bp).
    """
    # F1 short end (cash rate + BAB; OIS columns optional)
    f1_url = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
    try:
        f1_text = _http_get(f1_url, headers=_BROWSER_HEADERS)
    except Exception as exc:
        logger.debug("RBA F1 fetch (spread) failed: %s", exc)
        return None
    f1_short = _parse_rba_csv(f1_text, date_str)  # default = RBA_SERIES_MAP
    f1_ois = _rba_f1_ois_quotes(date_str, csv_text=f1_text)

    cash_rate = f1_short.get(1.0 / 365.0)
    bab_90d = f1_short.get(0.25)
    if cash_rate is None or bab_90d is None:
        return None

    # F2 govt yields
    f2 = _rba_f2_govt_yields(date_str)
    cgs_2y = f2.get(2.0)
    if cgs_2y is None:
        return None

    # Pick the OIS anchor: prefer 6M OIS if present (longest available),
    # else fall back to 90D BAB.  6M is closer to 1Y than 90D.
    ois_6m = f1_ois.get(0.5)
    ois_3m = f1_ois.get(0.25)
    if ois_6m is not None:
        ois_anchor = ois_6m
        ois_tenor = 0.5
    elif ois_3m is not None:
        ois_anchor = ois_3m
        ois_tenor = 0.25
    else:
        ois_anchor = bab_90d
        ois_tenor = 0.25

    # CGS at the same tenor: linear interp between cash rate (≈0y) and 2Y CGS.
    # cgs(t) = cash_rate + (cgs_2y - cash_rate) * (t / 2.0)
    cgs_at_tenor = cash_rate + (cgs_2y - cash_rate) * (ois_tenor / 2.0)
    spread = ois_anchor - cgs_at_tenor

    # Sanity check -- the OIS-CGS basis is normally within +/-100bp.
    if abs(spread) >= 0.020:  # 200 bp
        logger.debug("OIS-CGS spread %.4f outside sanity band; treating as None", spread)
        return None
    return float(spread)


def _rba_aud_ois(date_str: str) -> dict[float, float]:
    """AUD term structure: RBA F1 short end + F2 CGS-derived bridge + FRED long-end.

    Layers
    ------
    Tier-1a -- F1 short end (cash rate, 90D BAB, and 1M/3M/6M OIS when published).
    Tier-1b -- F2 govt yields with an OIS-CGS spread adjustment to imply OIS
                rates at the 2Y / 3Y / (5Y) tenors that ASX IB futures don't
                cleanly cover.
    Tier-2  -- FRED Australian series (3M interbank, 10Y govt) as fillers for
                tenors not supplied above.

    Result
    ------
    A dict with up to ~6 tenors covering overnight, 0.25, 0.5, 2.0, 3.0,
    and 10.0 -- substantially richer than the prior 3-tenor build.
    """
    out: dict[float, float] = {}

    # Tier 1a: RBA F1 daily CSV (cash rate + 90D BAB; reuse text for OIS)
    f1_url = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
    f1_text: Optional[str] = None
    try:
        f1_text = _http_get(f1_url, headers=_BROWSER_HEADERS)
        out.update(_parse_rba_csv(f1_text, date_str))
    except Exception as exc:
        logger.warning("RBA F1 fetch failed: %s", exc)

    # F1 OIS columns (1M/3M/6M) -- override BAB at 0.25 when 3M OIS is present.
    if f1_text is not None:
        f1_ois = _rba_f1_ois_quotes(date_str, csv_text=f1_text)
        for tenor, rate in f1_ois.items():
            out[tenor] = rate  # OIS preferred over BAB at overlapping tenor

    # Tier 1b: F2 CGS yields - OIS-CGS spread (bridge points at 2Y / 3Y / 5Y).
    try:
        f2_yields = _rba_f2_govt_yields(date_str)
    except Exception as exc:
        logger.warning("RBA F2 fetch failed: %s", exc)
        f2_yields = {}

    if f2_yields:
        spread = _aud_ois_cgs_spread(date_str)
        if spread is None:
            # Without a spread we can't strictly imply OIS from CGS, but the
            # CGS yield itself is a reasonable proxy for the OIS curve at
            # those tenors (basis usually small).  Use spread = 0 as a soft
            # fallback so we still get tenor coverage.
            spread = 0.0
        # Implied OIS = CGS + (OIS - CGS basis).  The basis is computed at a
        # comparable short tenor and assumed roughly flat across the 2Y-5Y
        # bridge.  We restrict the bridge to the 2Y / 3Y / 5Y tenors and
        # let FRED IRLTLT01AUM156N supply the 10Y point (already validated
        # as the existing long-end anchor).
        bridge_tenors = (2.0, 3.0, 5.0)
        for tenor in bridge_tenors:
            cgs_yield = f2_yields.get(tenor)
            if cgs_yield is None:
                continue
            out[tenor] = cgs_yield + spread

    # Tier 2: FRED fillers -- only used if no value yet at that tenor.
    fred_extra = {
        "IR3TIB01AUM156N": 0.25,    # 3M interbank
        "IRLTLT01AUM156N": 10.0,    # 10Y govt
    }
    for series_id, tenor in fred_extra.items():
        if tenor in out:
            continue
        try:
            rate = _fred_lookup(series_id, date_str)
        except Exception as exc:
            logger.warning("FRED %s failed: %s", series_id, exc)
            continue
        if rate is not None:
            out[tenor] = rate
    return out


def _fred_aud_proxy(date_str: str) -> dict[float, float]:
    """Tier 2 AUD: 2-point curve from FRED Australian series.

    Tenors:
        0.25 (3M) -> IR3TIB01AUM156N (AUS 3M interbank, monthly)
        10.0      -> IRLTLT01AUM156N (AUS 10Y govt, monthly)
    """
    api_key = config.get_fred_api_key()
    if not api_key:
        return {}

    plan = [
        ("IR3TIB01AUM156N", 0.25),
        ("IRLTLT01AUM156N", 10.0),
    ]
    curve: dict[float, float] = {}
    for series_id, tenor in plan:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&file_type=json&sort_order=desc&limit=5"
            f"&observation_end={date_str}"
            f"&api_key={api_key}"
        )
        try:
            json_text = _http_get(url)
        except Exception as exc:
            logger.debug("FRED AUD fetch %s failed: %s", series_id, exc)
            continue
        rate = _parse_fred_json(json_text)
        if rate is None:
            continue
        curve[tenor] = rate
    return curve


# ---------------------------------------------------------------------------
# Multi-tier dispatchers
# ---------------------------------------------------------------------------
def _eur_curve(date_str: str) -> dict[float, float]:
    """EUR multi-tier: ECB legacy SDMX -> FRED euro proxy -> empty."""
    try:
        curve = _ecb_sdw_eur_ois(date_str)
    except Exception as exc:
        logger.debug("EUR Tier 1 (ECB) errored: %s", exc)
        curve = {}
    if curve:
        logger.info("EUR curve from ECB legacy SDMX (%d tenors)", len(curve))
        return curve
    try:
        curve = _fred_eur_proxy(date_str)
    except Exception as exc:
        logger.debug("EUR Tier 2 (FRED) errored: %s", exc)
        curve = {}
    if curve:
        logger.info("EUR curve from FRED proxy (%d tenors)", len(curve))
    return curve


def _gbp_curve(date_str: str) -> dict[float, float]:
    """GBP multi-tier: BoE IADB -> FRED GBP proxy -> empty."""
    try:
        curve = _boe_iadb_gbp_ois(date_str)
    except Exception as exc:
        logger.debug("GBP Tier 1 (BoE) errored: %s", exc)
        curve = {}
    if curve:
        logger.info("GBP curve from BoE IADB (%d tenors)", len(curve))
        return curve
    try:
        curve = _fred_gbp_proxy(date_str)
    except Exception as exc:
        logger.debug("GBP Tier 2 (FRED) errored: %s", exc)
        curve = {}
    if curve:
        logger.info("GBP curve from FRED proxy (%d tenors)", len(curve))
    return curve


def _aud_curve(date_str: str) -> dict[float, float]:
    """AUD multi-tier: RBA F1 -> FRED AUD proxy -> empty."""
    try:
        curve = _rba_aud_ois(date_str)
    except Exception as exc:
        logger.debug("AUD Tier 1 (RBA) errored: %s", exc)
        curve = {}
    if curve:
        logger.info("AUD curve from RBA F1 (%d tenors)", len(curve))
        return curve
    try:
        curve = _fred_aud_proxy(date_str)
    except Exception as exc:
        logger.debug("AUD Tier 2 (FRED) errored: %s", exc)
        curve = {}
    if curve:
        logger.info("AUD curve from FRED proxy (%d tenors)", len(curve))
    return curve


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(currency: str, date_str: str) -> Path:
    return CACHE_DIR / f"{currency}_{date_str}.parquet"


def _read_cache(currency: str, date_str: str) -> Optional[dict[float, float]]:
    path = _cache_path(currency, date_str)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.debug("Cache read failed for %s: %s", path, exc)
        return None
    if "tenor_years" not in df.columns or "zero_rate" not in df.columns:
        return None
    return {float(t): float(r) for t, r in zip(df["tenor_years"], df["zero_rate"])}


def _write_cache(currency: str, date_str: str, curve: dict[float, float]) -> None:
    if not curve:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tenors = sorted(curve.keys())
    df = pd.DataFrame({
        "tenor_years": np.array(tenors, dtype=float),
        "zero_rate": np.array([curve[t] for t in tenors], dtype=float),
    })
    path = _cache_path(currency, date_str)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        logger.debug("Cache write failed for %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def fetch_foreign_ois_curve(currency: str, date_str: str) -> dict[float, float]:
    """Return {tenor_years: zero_rate_decimal} for one supported currency.

    Parameters
    ----------
    currency : str
        One of "EUR", "GBP", "JPY", "AUD".  Anything else raises ValueError.
    date_str : str
        Valuation date in ISO format ("YYYY-MM-DD").

    Returns
    -------
    dict[float, float]
        Term-structure mapping tenor (years) to zero rate (decimal).
        Empty dict on total failure.
    """
    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Unsupported currency '{currency}'. "
            f"Must be one of {SUPPORTED_CURRENCIES}."
        )

    cached = _read_cache(currency, date_str)
    if cached is not None:
        return cached

    try:
        if currency == "EUR":
            curve = _eur_curve(date_str)
        elif currency == "GBP":
            curve = _gbp_curve(date_str)
        elif currency == "JPY":
            curve = _boj_tona_jpy_ois(date_str)
        else:  # AUD
            curve = _aud_curve(date_str)
    except Exception as exc:
        logger.debug("Fetch failed for %s on %s: %s", currency, date_str, exc)
        return {}

    if curve:
        _write_cache(currency, date_str, curve)
    return curve
