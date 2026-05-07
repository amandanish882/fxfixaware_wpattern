"""
Tests for shared/exchange_rfr_loader.py
=======================================

The two public fetchers (``fetch_tona_strip`` and ``fetch_aonia_ib_strip``)
talk to live exchange websites in production via a multi-source try-chain.
The tests below mock the per-source helpers (or the underlying HTTP fetch)
with ``unittest.mock.patch`` and feed canned HTML/CSV/DF fixtures, so no
live network calls are made.

Run with:  pytest shared/tests/test_exchange_rfr_loader.py -v
"""

import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup so imports resolve from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import pytest

from shared import exchange_rfr_loader as erl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_TONA_HTML_FIXTURE = """
<html><body>
<h2>3M TONA Futures Final Settlement Prices</h2>
<table>
  <thead>
    <tr><th>Contract Month</th><th>Settlement Price</th></tr>
  </thead>
  <tbody>
    <tr><td>TONA Jun 2026</td><td>99.500</td></tr>
    <tr><td>TONA Sep 2026</td><td>99.450</td></tr>
    <tr><td>TONA Dec 2026</td><td>99.400</td></tr>
    <tr><td>TONA Mar 2027</td><td>99.350</td></tr>
  </tbody>
</table>
</body></html>
"""

# JPX daily-settlement CSV fixture: 2-row preamble + header + a few TONA
# rows + an unrelated row.  Issue-name suffix is YYMMDD = the last
# trading day, which is the business day before the 3rd Wednesday of
# the IMM month.
#   Jun 2026 IMM = 2026-06-17 -> LTD = 2026-06-16
#   Sep 2026 IMM = 2026-09-16 -> LTD = 2026-09-15
#   Dec 2026 IMM = 2026-12-16 -> LTD = 2026-12-15
#   Mar 2027 IMM = 2027-03-17 -> LTD = 2027-03-16
_TONA_CSV_FIXTURE = (
    "Sorted by ...,,,,,,,,,,,\n"
    "Issues passed the trading date are not listed.,,,,,,,,,,,\n"
    "Issue Code,Issue Name,Put/Call,Contract Month,Strike Price,"
    "Settlement Price,Theoretical Price,Underlying Price,Volatility,"
    "Interest Rate,Days until Maturity,Underlying Name\n"
    "161030091,FUT_TOA3M_260616,,202603,,99.500,,,,,49,3-Month TONA\n"
    "161060091,FUT_TOA3M_260915,,202606,,99.450,,,,,140,3-Month TONA\n"
    "161090091,FUT_TOA3M_261215,,202609,,99.400,,,,,231,3-Month TONA\n"
    "161120091,FUT_TOA3M_270316,,202612,,99.350,,,,,322,3-Month TONA\n"
    "161060018,FUT_225_260611,,202606,,59420,59574,59513.12,,0.8944,42,Nikkei 225\n"
)

_JPX_INDEX_FIXTURE = """
<html><body>
<h2>Settlement Prices</h2>
<table><tr><td>As of (May. 01, 2026)</td><td></td></tr></table>
<a href="/english/markets/derivatives/settlement-price/tvdivq00000014l6-att/rb_e20260501.csv">
Daily settlement CSV
</a>
</body></html>
"""

_ASX_HTML_FIXTURE = """
<html><body>
<h2>30 Day Interbank Cash Rate Futures</h2>
<table>
  <thead>
    <tr><th>Contract</th><th>Settlement Price</th></tr>
  </thead>
  <tbody>
    <tr><td>IBM6 (Jun 26)</td><td>96.200</td></tr>
    <tr><td>IBN6 (Jul 26)</td><td>96.250</td></tr>
    <tr><td>IBQ6 (Aug 26)</td><td>96.300</td></tr>
    <tr><td>IBU6 (Sep 26)</td><td>96.350</td></tr>
  </tbody>
</table>
</body></html>
"""

# Markit research API JSON response (a faithful trim of the live shape).
_ASX_API_FIXTURE = """{
    "data": {
        "items": [
            {"symbol": "IBK2026", "dateExpiry": "2026-05-27",
             "priceLastTrade": 95.69,
             "pricePreviousSettlement": 95.695},
            {"symbol": "IBM2026", "dateExpiry": "2026-06-28",
             "priceLastTrade": 95.625,
             "pricePreviousSettlement": 95.630},
            {"symbol": "IBN2026", "dateExpiry": "2026-07-29",
             "priceLastTrade": 95.595,
             "pricePreviousSettlement": 95.610},
            {"symbol": "IBQ2026", "dateExpiry": "2026-08-29",
             "priceLastTrade": 95.505,
             "pricePreviousSettlement": 95.520},
            {"symbol": "IBU2026", "dateExpiry": "2026-09-28",
             "priceLastTrade": 95.45,
             "pricePreviousSettlement": 95.470},
            {"symbol": "IBV2026", "dateExpiry": "2026-10-28",
             "priceLastTrade": 95.38,
             "pricePreviousSettlement": 95.395}
        ],
        "chart": {"svg": ""}
    }
}"""

_VAL_DATE = "2026-04-28"


# ---------------------------------------------------------------------------
# Schema / empty-fallback contract
# ---------------------------------------------------------------------------
def test_empty_rfr_df_schema():
    df = erl._empty_rfr_df()
    assert list(df.columns) == ["contract", "expiry_years", "price",
                                "implied_rate"]
    assert df.empty


def test_fetch_tona_strip_returns_empty_on_http_failure(tmp_path, monkeypatch):
    """If every HTTP fetch raises, the function must NOT raise -- it must
    return an empty DataFrame with the right schema."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    def boom(*_a, **_kw):
        raise OSError("network unreachable (simulated)")

    # Patch the low-level HTTP getter (covers all URL-based sources) AND
    # the investpy fallback so it can't reach the real network either.
    with patch.object(erl, "_http_get", side_effect=boom), \
         patch.object(erl, "_try_investpy_tona", return_value=erl._empty_rfr_df()):
        out = erl.fetch_tona_strip(_VAL_DATE)

    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_fetch_aonia_ib_strip_returns_empty_on_http_failure(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    def boom(*_a, **_kw):
        raise OSError("network unreachable (simulated)")

    # Patch _http_get (covers both the JSON API and the HTML scrape)
    # AND the investpy fallback so it can't reach the real network either.
    with patch.object(erl, "_http_get", side_effect=boom), \
         patch.object(
             erl, "_try_investpy_aonia", return_value=erl._empty_rfr_df(),
         ):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)

    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_fetch_tona_strip_invalid_date_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    out = erl.fetch_tona_strip("not-a-date")
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_fetch_aonia_ib_strip_invalid_date_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    out = erl.fetch_aonia_ib_strip("nope")
    assert out.empty


# ---------------------------------------------------------------------------
# Mocked-source happy paths
# ---------------------------------------------------------------------------
def test_fetch_tona_strip_parses_jpx_csv(tmp_path, monkeypatch):
    """Feed a fake JPX index page + CSV via _http_get and assert the
    daily-settlement CSV path produces the strip."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    def fake_get(url, timeout=15):
        if url.endswith(".csv"):
            return _TONA_CSV_FIXTURE
        return _JPX_INDEX_FIXTURE

    with patch.object(erl, "_http_get", side_effect=fake_get):
        out = erl.fetch_tona_strip(_VAL_DATE)

    assert not out.empty, "TONA strip should parse from the CSV fixture"
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]
    assert len(out) == 4

    # All expiries strictly positive and sorted ascending.
    assert (out["expiry_years"] > 0).all()
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all(), out["expiry_years"].tolist()

    # implied_rate = (100 - price) / 100 within 1e-6.
    for _, r in out.iterrows():
        expected = (100.0 - r["price"]) / 100.0
        assert r["implied_rate"] == pytest.approx(expected, abs=1e-6)

    # First contract should be Jun 2026 -> IMM = 2026-06-17.
    val = dt.date.fromisoformat(_VAL_DATE)
    expected_jun = (dt.date(2026, 6, 17) - val).days / 365.25
    assert out["expiry_years"].iloc[0] == pytest.approx(expected_jun,
                                                        abs=1e-6)
    assert out["contract"].iloc[0] == "TYM6"


def test_fetch_tona_strip_falls_back_to_sq_html(tmp_path, monkeypatch):
    """If the JPX-CSV source returns empty, the SQ-HTML source kicks in."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    with patch.object(erl, "_try_jpx_settlement_csv",
                      return_value=erl._empty_rfr_df()), \
         patch.object(erl, "_http_get", return_value=_TONA_HTML_FIXTURE):
        out = erl.fetch_tona_strip(_VAL_DATE)

    assert not out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]
    assert len(out) == 4
    assert out["contract"].iloc[0] == "TYM6"


def test_fetch_tona_strip_all_sources_fail_returns_empty(tmp_path, monkeypatch):
    """If every source returns empty, the function returns empty -- no raise."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    empty = erl._empty_rfr_df()
    with patch.object(erl, "_try_jpx_settlement_csv", return_value=empty), \
         patch.object(erl, "_try_jpx_sq_html", return_value=empty), \
         patch.object(erl, "_try_investpy_tona", return_value=empty):
        out = erl.fetch_tona_strip(_VAL_DATE)

    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_fetch_aonia_ib_strip_parses_mocked_html(tmp_path, monkeypatch):
    """The API source returns empty (HTML body fails JSON parse), so the
    HTML scraper takes over and produces the strip."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    with patch.object(erl, "_try_asx_api",
                      return_value=erl._empty_rfr_df()), \
         patch.object(erl, "_http_get", return_value=_ASX_HTML_FIXTURE):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)

    assert not out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]
    assert len(out) == 4

    assert (out["expiry_years"] > 0).all()
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all()

    for _, r in out.iterrows():
        expected = (100.0 - r["price"]) / 100.0
        assert r["implied_rate"] == pytest.approx(expected, abs=1e-6)

    # IBM6 = Jun 2026, anchored to month-end = 2026-06-30.
    val = dt.date.fromisoformat(_VAL_DATE)
    by_contract = dict(zip(out["contract"], out["expiry_years"]))
    expected_m6 = (dt.date(2026, 6, 30) - val).days / 365.25
    assert by_contract["IBM6"] == pytest.approx(expected_m6, abs=1e-6)


def test_fetch_aonia_ib_strip_falls_back_to_investpy(tmp_path, monkeypatch):
    """If both the API and HTML sources return empty, the investpy
    source kicks in."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    fake_strip = pd.DataFrame(
        [{
            "contract": "IBK6",  # May 2026 anchored to month-end
            "expiry_years": 0.10,
            "price": 95.755,
            "implied_rate": 0.04245,
        }],
        columns=erl._RFR_COLS,
    )
    empty = erl._empty_rfr_df()
    with patch.object(erl, "_try_asx_api", return_value=empty), \
         patch.object(erl, "_try_asx_html", return_value=empty), \
         patch.object(erl, "_try_investpy_aonia", return_value=fake_strip):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)

    assert not out.empty
    assert len(out) == 1
    assert out["contract"].iloc[0] == "IBK6"
    assert out["price"].iloc[0] == pytest.approx(95.755, abs=1e-6)


def test_fetch_aonia_ib_strip_all_sources_fail_returns_empty(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    empty = erl._empty_rfr_df()
    with patch.object(erl, "_try_asx_api", return_value=empty), \
         patch.object(erl, "_try_asx_html", return_value=empty), \
         patch.object(erl, "_try_investpy_aonia", return_value=empty):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_fetch_aonia_ib_strip_parses_api_json(tmp_path, monkeypatch):
    """The Markit JSON API source is the first to be tried, and a
    well-formed payload should produce the full strip without ever
    falling through to the HTML scrape."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    captured: list[str] = []

    def fake_get(url, timeout=15):
        captured.append(url)
        return _ASX_API_FIXTURE

    with patch.object(erl, "_http_get", side_effect=fake_get):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)

    # All 6 contracts in the fixture should round-trip.
    assert not out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]
    assert len(out) == 6
    # Only one URL was hit -- the API.  The HTML source was never reached.
    assert len(captured) == 1
    assert "markitdigital" in captured[0]

    # All expiries strictly positive and monotonically increasing.
    assert (out["expiry_years"] > 0).all()
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all(), out["expiry_years"].tolist()

    # implied_rate = (100 - price) / 100 exactly.
    for _, r in out.iterrows():
        expected = (100.0 - r["price"]) / 100.0
        assert r["implied_rate"] == pytest.approx(expected, abs=1e-6)

    # Settlement preferred over last-trade.
    rates = dict(zip(out["contract"], out["price"]))
    assert rates["IBK6"] == pytest.approx(95.695, abs=1e-9)
    assert rates["IBM6"] == pytest.approx(95.630, abs=1e-9)

    # IBK6 = May 2026, anchored to month-end = 2026-05-31.
    val = dt.date.fromisoformat(_VAL_DATE)
    by_contract = dict(zip(out["contract"], out["expiry_years"]))
    expected_k6 = (dt.date(2026, 5, 31) - val).days / 365.25
    assert by_contract["IBK6"] == pytest.approx(expected_k6, abs=1e-6)


def test_fetch_aonia_ib_strip_api_then_html(tmp_path, monkeypatch):
    """Mock the API source returning empty + the HTML source returning
    rows -- assert the HTML rows are what comes back."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    with patch.object(erl, "_try_asx_api",
                      return_value=erl._empty_rfr_df()), \
         patch.object(erl, "_http_get", return_value=_ASX_HTML_FIXTURE):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)

    assert not out.empty
    assert len(out) == 4
    assert set(out["contract"]) == {"IBM6", "IBN6", "IBQ6", "IBU6"}


def test_fetch_aonia_ib_strip_api_bad_json_returns_empty(tmp_path, monkeypatch):
    """If the API source returns non-JSON garbage, _try_asx_api logs and
    yields an empty DF -- it must NOT raise."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    val = dt.date.fromisoformat(_VAL_DATE)
    with patch.object(erl, "_http_get", return_value="not-json-at-all"):
        out = erl._try_asx_api(val)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_parse_asx_api_payload_skips_past_expiries():
    """Contracts whose contract-month-end has already lapsed must be
    filtered out (expiry_years <= 0)."""
    val = dt.date(2026, 5, 15)
    payload = {
        "data": {
            "items": [
                # April 2026 month-end (2026-04-30) is already in the
                # past relative to val_date -- should be dropped.
                {"symbol": "IBJ2026", "dateExpiry": "2026-04-28",
                 "pricePreviousSettlement": 95.5},
                # June 2026 month-end is in the future -- should be kept.
                {"symbol": "IBM2026", "dateExpiry": "2026-06-28",
                 "pricePreviousSettlement": 95.7},
            ],
        },
    }
    out = erl._parse_asx_api_payload(payload, val)
    assert len(out) == 1
    assert out["contract"].iloc[0] == "IBM6"


def test_fetch_aonia_ib_strip_18_contracts_smoke(tmp_path, monkeypatch):
    """Synthesise a full 18-contract feed and assert all 18 round-trip
    monotonic and with the right schema."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    val = dt.date.fromisoformat(_VAL_DATE)
    items = []
    year, month = val.year, val.month
    for _ in range(18):
        # advance one calendar month and pick the 28th as the expiry day.
        year, month = erl._add_months(year, month, 1)
        code = erl._MONTH_CODE_BY_MONTH[month]
        items.append({
            "symbol": f"IB{code}{year}",
            "dateExpiry": f"{year:04d}-{month:02d}-28",
            "priceLastTrade": 95.5,
            "pricePreviousSettlement": 95.5,
        })
    import json as _json
    payload_text = _json.dumps({"data": {"items": items}})

    with patch.object(erl, "_http_get", return_value=payload_text):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)

    assert len(out) == 18
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs > 0).all()


def test_fetch_tona_strip_implied_rate_decimal(tmp_path, monkeypatch):
    """price=99.50 -> implied_rate=0.0050 etc."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    def fake_get(url, timeout=15):
        if url.endswith(".csv"):
            return _TONA_CSV_FIXTURE
        return _JPX_INDEX_FIXTURE

    with patch.object(erl, "_http_get", side_effect=fake_get):
        out = erl.fetch_tona_strip(_VAL_DATE)
    rates = dict(zip(out["contract"], out["implied_rate"]))
    assert rates["TYM6"] == pytest.approx(0.0050, abs=1e-9)
    assert rates["TYU6"] == pytest.approx(0.0055, abs=1e-9)


def test_fetch_aonia_ib_strip_implied_rate_decimal(tmp_path, monkeypatch):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    with patch.object(erl, "_try_asx_api",
                      return_value=erl._empty_rfr_df()), \
         patch.object(erl, "_http_get", return_value=_ASX_HTML_FIXTURE):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)
    rates = dict(zip(out["contract"], out["implied_rate"]))
    assert rates["IBM6"] == pytest.approx(0.0380, abs=1e-9)
    assert rates["IBN6"] == pytest.approx(0.0375, abs=1e-9)


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------
def test_fetch_tona_strip_uses_cache(tmp_path, monkeypatch):
    """Second call within the TTL should NOT hit any source again."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    def fake_get(url, timeout=15):
        if url.endswith(".csv"):
            return _TONA_CSV_FIXTURE
        return _JPX_INDEX_FIXTURE

    with patch.object(erl, "_http_get", side_effect=fake_get) as mock_get:
        first = erl.fetch_tona_strip(_VAL_DATE)
        second = erl.fetch_tona_strip(_VAL_DATE)

    assert not first.empty and not second.empty
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
        check_dtype=False,
    )
    # Two calls expected on the first fetch (index page + CSV); zero on
    # the second (served from cache).
    assert mock_get.call_count == 2, (
        f"expected exactly 2 HTTP calls (index + CSV, then cache), "
        f"got {mock_get.call_count}"
    )


def test_fetch_aonia_ib_strip_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)

    with patch.object(erl, "_try_asx_api",
                      return_value=erl._empty_rfr_df()), \
         patch.object(erl, "_http_get",
                      return_value=_ASX_HTML_FIXTURE) as mock_get:
        first = erl.fetch_aonia_ib_strip(_VAL_DATE)
        second = erl.fetch_aonia_ib_strip(_VAL_DATE)

    assert not first.empty and not second.empty
    # First call hits HTML once; second call is served entirely from cache.
    assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_third_wednesday_helper():
    assert erl._third_wednesday(2026, 6) == dt.date(2026, 6, 17)
    assert erl._third_wednesday(2026, 9) == dt.date(2026, 9, 16)
    assert erl._third_wednesday(2026, 12) == dt.date(2026, 12, 16)
    assert erl._third_wednesday(2027, 3) == dt.date(2027, 3, 17)


def test_month_end_helper():
    assert erl._month_end(2026, 6) == dt.date(2026, 6, 30)
    assert erl._month_end(2026, 2) == dt.date(2026, 2, 28)
    assert erl._month_end(2024, 2) == dt.date(2024, 2, 29)  # leap
    assert erl._month_end(2026, 12) == dt.date(2026, 12, 31)


def test_add_months_helper():
    assert erl._add_months(2026, 4, 1) == (2026, 5)
    assert erl._add_months(2026, 12, 1) == (2027, 1)
    assert erl._add_months(2026, 11, 3) == (2027, 2)


def test_parse_jpx_contract_label():
    assert erl._parse_jpx_contract_label("TONA Jun 2026") == (2026, 6)
    assert erl._parse_jpx_contract_label("TONA Sep-26") == (2026, 9)
    assert erl._parse_jpx_contract_label("3M TONA Mar 2027") == (2027, 3)
    assert erl._parse_jpx_contract_label("nothing here") is None
    assert erl._parse_jpx_contract_label(None) is None  # type: ignore[arg-type]


def test_find_jpx_settlement_csv_url():
    url = erl._find_jpx_settlement_csv_url(_JPX_INDEX_FIXTURE)
    assert url is not None
    assert url.endswith("rb_e20260501.csv")
    assert url.startswith("https://www.jpx.co.jp/")


def test_find_jpx_settlement_csv_url_no_match():
    assert erl._find_jpx_settlement_csv_url("<html>no link</html>") is None


def test_parse_asx_contract_label_compact():
    val = dt.date.fromisoformat(_VAL_DATE)
    out = erl._parse_asx_contract_label("IBM6", val)
    assert out is not None
    symbol, year, month = out
    assert symbol == "IBM6"
    assert year == 2026
    assert month == 6


def test_parse_asx_contract_label_long():
    val = dt.date.fromisoformat(_VAL_DATE)
    out = erl._parse_asx_contract_label("Jun 26", val)
    assert out is not None
    symbol, year, month = out
    assert symbol == "IBM6"
    assert year == 2026
    assert month == 6


def test_parse_asx_contract_label_no_match():
    val = dt.date.fromisoformat(_VAL_DATE)
    assert erl._parse_asx_contract_label("not a contract", val) is None


# ---------------------------------------------------------------------------
# Empty-table fallback
# ---------------------------------------------------------------------------
def test_fetch_tona_strip_empty_html_returns_empty(tmp_path, monkeypatch):
    """Pages with no TONA tables AND no CSV link AND no investpy data
    -> empty DataFrame, not a raise."""
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    with patch.object(erl, "_http_get",
                      return_value="<html><body>No data</body></html>"), \
         patch.object(erl, "_try_investpy_tona",
                      return_value=erl._empty_rfr_df()):
        out = erl.fetch_tona_strip(_VAL_DATE)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price",
                                 "implied_rate"]


def test_fetch_aonia_ib_strip_empty_html_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(erl, "CACHE_DIR", tmp_path)
    # _http_get returns the same dummy HTML for both the API and the
    # HTML scrape; the JSON parse fails, the HTML has no IB tables.
    with patch.object(erl, "_http_get",
                      return_value="<html><body>No data</body></html>"), \
         patch.object(erl, "_try_investpy_aonia",
                      return_value=erl._empty_rfr_df()):
        out = erl.fetch_aonia_ib_strip(_VAL_DATE)
    assert out.empty
