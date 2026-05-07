"""
Tests for shared/databento_curve_loader.py
==========================================

Verifies that the parquet -> curve DataFrame transforms behave as expected
for both the FX back-month curve and the SOFR (SR3) strip.  All cache
locations are redirected via ``monkeypatch`` so the tests do not depend on
the real ``data/databento_cache/`` directory.

Run with:  pytest shared/tests/test_databento_curve_loader.py -v
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so imports resolve from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import pytest

from shared import databento_curve_loader as dcl


# ---------------------------------------------------------------------------
# FX back-month curve
# ---------------------------------------------------------------------------
def test_fetch_back_month_fx_curve_filters_to_ticker(tmp_path, monkeypatch):
    """Only rows whose symbol starts with the requested ticker are returned,
    expiry_years are positive, and the curve is sorted ascending."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    rows = pd.DataFrame({
        "ts_event": pd.to_datetime([
            f"{date_str} 21:00:00",
            f"{date_str} 21:00:00",
            f"{date_str} 21:00:00",
            f"{date_str} 21:00:00",
        ], utc=True),
        "symbol": ["6EM6.c.0", "6EU6.c.0", "6EZ6.c.0", "6BM6.c.0"],
        "close": [1.0850, 1.0875, 1.0900, 1.2700],
        "open":  [1.0840, 1.0860, 1.0890, 1.2690],
        "high":  [1.0860, 1.0880, 1.0905, 1.2710],
        "low":   [1.0830, 1.0855, 1.0885, 1.2685],
        "volume": [1000, 800, 500, 600],
    })
    cache_path = tmp_path / f"fx_back_month_curve_{date_str}.parquet"
    rows.to_parquet(cache_path, index=False)

    out = dcl.fetch_back_month_fx_curve("6E", date_str, n_contracts=4)

    assert list(out.columns) == ["contract", "expiry_years", "price"]
    assert len(out) == 3, f"expected 3 6E rows, got {len(out)}"

    # Only 6E* symbols
    assert all(c.startswith("6E") for c in out["contract"]), out["contract"].tolist()
    assert "6BM6.c.0" not in out["contract"].tolist()

    # All expiries strictly positive
    assert (out["expiry_years"] > 0).all()

    # Ascending in expiry_years
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all(), f"not monotonic: {out['expiry_years'].tolist()}"


def test_fetch_back_month_fx_curve_missing_cache_returns_empty(tmp_path, monkeypatch):
    """Missing cache file -> empty DataFrame with the right column schema."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    out = dcl.fetch_back_month_fx_curve("6E", "2099-01-01", n_contracts=4)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price"]


def test_fetch_back_month_fx_curve_takes_last_close_per_symbol(tmp_path, monkeypatch):
    """Multiple intra-day rows -> the LAST close per symbol is used."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    rows = pd.DataFrame({
        "ts_event": pd.to_datetime([
            f"{date_str} 13:00:00",
            f"{date_str} 21:00:00",  # later -> should win
            f"{date_str} 13:00:00",
        ], utc=True),
        "symbol": ["6EM6.c.0", "6EM6.c.0", "6EU6.c.0"],
        "close": [1.0800, 1.0888, 1.0875],
        "open":  [1.0795, 1.0880, 1.0860],
        "high":  [1.0810, 1.0890, 1.0880],
        "low":   [1.0790, 1.0870, 1.0855],
        "volume": [500, 1000, 700],
    })
    cache_path = tmp_path / f"fx_back_month_curve_{date_str}.parquet"
    rows.to_parquet(cache_path, index=False)

    out = dcl.fetch_back_month_fx_curve("6E", date_str, n_contracts=4)

    em6 = out[out["contract"] == "6EM6.c.0"]
    assert len(em6) == 1
    assert em6["price"].iloc[0] == pytest.approx(1.0888)


# ---------------------------------------------------------------------------
# SOFR strip
# ---------------------------------------------------------------------------
def test_fetch_sofr_strip_computes_implied_rate(tmp_path, monkeypatch):
    """implied_rate column equals (100 - price) / 100."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    rows = pd.DataFrame({
        "ts_event": pd.to_datetime([
            f"{date_str} 21:00:00",
            f"{date_str} 21:00:00",
        ], utc=True),
        "symbol": ["SR3M6.c.0", "SR3U6.c.0"],
        "close": [96.50, 96.75],
        "open":  [96.45, 96.70],
        "high":  [96.55, 96.80],
        "low":   [96.40, 96.65],
        "volume": [10000, 8000],
    })
    cache_path = tmp_path / f"sofr_sr3_strip_1y_to_{date_str}.parquet"
    rows.to_parquet(cache_path, index=False)

    out = dcl.fetch_sofr_strip(date_str, n_contracts=8)

    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    assert len(out) == 2

    # implied_rate = (100 - price) / 100, in decimal form
    for _, r in out.iterrows():
        expected = (100.0 - r["price"]) / 100.0
        assert r["implied_rate"] == pytest.approx(expected)

    # Specifically: 96.50 -> 0.0350, 96.75 -> 0.0325
    rates_by_contract = dict(zip(out["contract"], out["implied_rate"]))
    assert rates_by_contract["SR3M6.c.0"] == pytest.approx(0.0350)
    assert rates_by_contract["SR3U6.c.0"] == pytest.approx(0.0325)

    # Curve sorted ascending in expiry_years
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all()


def test_fetch_sofr_strip_missing_cache_returns_empty(tmp_path, monkeypatch):
    """Missing cache file -> empty DataFrame with the right column schema."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    out = dcl.fetch_sofr_strip("2099-01-01", n_contracts=8)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]


# ---------------------------------------------------------------------------
# SR3 symbol parser
# ---------------------------------------------------------------------------
def test_parse_sr3_continuous_symbol_format():
    """Cached SR3 strip uses bare-root continuous symbols like ``SR3.c.0``."""
    import datetime as dt

    val_date = dt.date.fromisoformat("2026-04-28")

    # Front contract: SR3.c.0 -> next quarterly on/after 2026-04-28 is June (M6).
    out = dcl._parse_sr3_symbol("SR3.c.0", val_date=val_date)
    assert out is not None, "should parse SR3.c.0"
    month_code, year_digit = out
    assert month_code == "M", f"expected June (M), got {month_code}"
    assert year_digit == 6, f"expected year digit 6 (2026), got {year_digit}"

    # SR3.c.1 -> September 2026 (U6).
    out = dcl._parse_sr3_symbol("SR3.c.1", val_date=val_date)
    assert out == ("U", 6), out

    # SR3.c.4 -> June 2027 (M7) (4 quarters past front).
    out = dcl._parse_sr3_symbol("SR3.c.4", val_date=val_date)
    assert out == ("M", 7), out

    # SR3.c.7 -> March 2028 (H8) (7 quarters past front).
    out = dcl._parse_sr3_symbol("SR3.c.7", val_date=val_date)
    assert out == ("H", 8), out


def test_parse_sr3_explicit_symbol_format():
    """Explicit ``SR3M6.c.0`` style should still parse to (month, year_digit)."""
    out = dcl._parse_sr3_symbol("SR3M6.c.0")
    assert out == ("M", 6), out

    # Without continuous suffix.
    out = dcl._parse_sr3_symbol("SR3U7")
    assert out == ("U", 7), out


def test_parse_sr3_rejects_non_sr3_symbol():
    """Symbols not starting with SR3 return None."""
    assert dcl._parse_sr3_symbol("6EM6.c.0") is None
    assert dcl._parse_sr3_symbol("FOO") is None


def test_parse_sr3_bare_root_requires_val_date():
    """Bare-root continuous symbols cannot be resolved without a val_date."""
    assert dcl._parse_sr3_symbol("SR3.c.0") is None


def test_fetch_sofr_strip_resolves_bare_root_continuous(tmp_path, monkeypatch):
    """End-to-end: bare-root ``SR3.c.N`` symbols produce a non-empty strip."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    rows = pd.DataFrame({
        "ts_event": pd.to_datetime([
            f"{date_str} 00:00:00",
            f"{date_str} 00:00:00",
            f"{date_str} 00:00:00",
            f"{date_str} 00:00:00",
        ], utc=True),
        "symbol": ["SR3.c.0", "SR3.c.1", "SR3.c.2", "SR3.c.3"],
        "close": [96.50, 96.55, 96.60, 96.65],
        "open":  [96.45, 96.50, 96.55, 96.60],
        "high":  [96.55, 96.60, 96.65, 96.70],
        "low":   [96.40, 96.45, 96.50, 96.55],
        "volume": [10000, 9000, 8000, 7000],
    })
    cache_path = tmp_path / f"sofr_sr3_strip_1y_to_{date_str}.parquet"
    rows.to_parquet(cache_path, index=False)

    out = dcl.fetch_sofr_strip(date_str, n_contracts=8)

    assert not out.empty, "bare-root SR3.c.N symbols should now parse"
    assert len(out) == 4
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    # Strictly positive expiries, monotone non-decreasing.
    assert (out["expiry_years"] > 0).all()
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all(), out["expiry_years"].tolist()


def test_fetch_sofr_strip_expiry_years_is_imm_date(tmp_path, monkeypatch):
    """``expiry_years`` should be the time to the IMM date (3rd Wed of the
    contract month), NOT the time to the end of the SR3 reference period."""
    import datetime as dt

    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    rows = pd.DataFrame({
        "ts_event": pd.to_datetime([
            f"{date_str} 00:00:00",
            f"{date_str} 00:00:00",
        ], utc=True),
        # SR3M6 IMM = 3rd Wed of June 2026 = 2026-06-17
        # SR3U6 IMM = 3rd Wed of September 2026 = 2026-09-16
        "symbol": ["SR3M6.c.0", "SR3U6.c.0"],
        "close": [96.50, 96.75],
        "open":  [96.45, 96.70],
        "high":  [96.55, 96.80],
        "low":   [96.40, 96.65],
        "volume": [10000, 8000],
    })
    cache_path = tmp_path / f"sofr_sr3_strip_1y_to_{date_str}.parquet"
    rows.to_parquet(cache_path, index=False)

    out = dcl.fetch_sofr_strip(date_str, n_contracts=8)
    by_contract = dict(zip(out["contract"], out["expiry_years"]))

    val_date = dt.date.fromisoformat(date_str)
    expected_m6 = (dt.date(2026, 6, 17) - val_date).days / 365.25  # ~0.137
    expected_u6 = (dt.date(2026, 9, 16) - val_date).days / 365.25  # ~0.386
    assert by_contract["SR3M6.c.0"] == pytest.approx(expected_m6, abs=1e-6)
    assert by_contract["SR3U6.c.0"] == pytest.approx(expected_u6, abs=1e-6)
    # Sanity: well below the old ref-end convention (~0.334 for M6).
    assert by_contract["SR3M6.c.0"] < 0.20


def test_sr3_imm_date_helper_third_wednesday():
    """``_sr3_imm_date`` should resolve to the 3rd Wednesday of the contract month."""
    import datetime as dt

    # M6 -> June 2026, 3rd Wednesday = 2026-06-17
    assert dcl._sr3_imm_date("M", 2026) == dt.date(2026, 6, 17)
    # U6 -> September 2026, 3rd Wednesday = 2026-09-16
    assert dcl._sr3_imm_date("U", 2026) == dt.date(2026, 9, 16)
    # H7 -> March 2027, 3rd Wednesday = 2027-03-17
    assert dcl._sr3_imm_date("H", 2027) == dt.date(2027, 3, 17)
    # Z6 -> December 2026, 3rd Wednesday = 2026-12-16
    assert dcl._sr3_imm_date("Z", 2026) == dt.date(2026, 12, 16)


# ---------------------------------------------------------------------------
# Foreign-currency RFR strip helpers (shared infrastructure)
# ---------------------------------------------------------------------------
def _make_strip_cache(tmp_path, filename, root, n, val_date_str, base_price=96.0):
    """Helper: write a synthetic OHLCV-1d parquet with bare-root continuous
    symbols ``{root}.c.0..{n-1}``."""
    rows = pd.DataFrame({
        "ts_event": pd.to_datetime(
            [f"{val_date_str} 00:00:00"] * n, utc=True,
        ),
        "symbol": [f"{root}.c.{i}" for i in range(n)],
        "close":  [base_price + 0.05 * i for i in range(n)],
        "open":   [base_price - 0.01 + 0.05 * i for i in range(n)],
        "high":   [base_price + 0.05 + 0.05 * i for i in range(n)],
        "low":    [base_price - 0.05 + 0.05 * i for i in range(n)],
        "volume": [1000 * (n - i) for i in range(n)],
    })
    cache_path = tmp_path / filename
    rows.to_parquet(cache_path, index=False)
    return cache_path


def test_parse_continuous_index_basic():
    """``_parse_continuous_index`` extracts the integer N from ``{root}.c.{N}``."""
    assert dcl._parse_continuous_index("SR1.c.0", "SR1") == 0
    assert dcl._parse_continuous_index("SO3.c.7", "SO3") == 7
    assert dcl._parse_continuous_index("FEMP.c.5", "FEMP") == 5
    # Wrong root.
    assert dcl._parse_continuous_index("SR3.c.0", "SR1") is None
    # Non-continuous form.
    assert dcl._parse_continuous_index("SR1H6", "SR1") is None
    # Bad index.
    assert dcl._parse_continuous_index("SR1.c.x", "SR1") is None


def test_step_months_helper():
    """``_step_months`` walks calendar months with year carry."""
    assert dcl._step_months(4, 2026, 0) == (4, 2026)
    assert dcl._step_months(4, 2026, 1) == (5, 2026)
    assert dcl._step_months(4, 2026, 9) == (1, 2027)
    assert dcl._step_months(12, 2026, 1) == (1, 2027)


# ---------------------------------------------------------------------------
# SR1 (1M SOFR) strip
# ---------------------------------------------------------------------------
def test_fetch_sr1_strip_columns_and_rates(tmp_path, monkeypatch):
    """SR1 strip returns the right schema, monotone expiries, and correct rates."""
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"sofr_sr1_strip_1y_to_{date_str}.parquet",
        root="SR1", n=7, val_date_str=date_str, base_price=95.50,
    )

    out = dcl.fetch_sr1_strip(date_str, n_contracts=7)

    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    # SR1.c.0 maps to the 3rd Wed of April 2026 (= 2026-04-15) which is
    # already past relative to val_date 2026-04-28, so it is filtered out
    # and we expect the remaining 6 forward contracts.
    assert len(out) == 6

    # implied_rate = (100 - price) / 100
    for _, r in out.iterrows():
        assert r["implied_rate"] == pytest.approx((100.0 - r["price"]) / 100.0)

    # Strictly positive, monotone non-decreasing expiries.
    assert (out["expiry_years"] > 0).all()
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all(), out["expiry_years"].tolist()


def test_fetch_sr1_strip_expiry_uses_third_wednesday(tmp_path, monkeypatch):
    """SR1.c.0 -> ref_start = 3rd Wed of valuation month (April 2026 = 2026-04-15)."""
    import datetime as dt
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"sofr_sr1_strip_1y_to_{date_str}.parquet",
        root="SR1", n=2, val_date_str=date_str,
    )

    out = dcl.fetch_sr1_strip(date_str, n_contracts=7)
    # The 3rd Wed of April 2026 (2026-04-15) is BEFORE val_date 2026-04-28,
    # so SR1.c.0 should be filtered out (expiry_years <= 0).  SR1.c.1 should
    # land on the 3rd Wed of May 2026 = 2026-05-20.
    assert "SR1.c.0" not in out["contract"].tolist()
    assert "SR1.c.1" in out["contract"].tolist()

    val_date = dt.date.fromisoformat(date_str)
    expected = (dt.date(2026, 5, 20) - val_date).days / 365.25
    row = out[out["contract"] == "SR1.c.1"].iloc[0]
    assert row["expiry_years"] == pytest.approx(expected, abs=1e-6)


def test_fetch_sr1_strip_missing_cache_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)
    out = dcl.fetch_sr1_strip("2099-01-01", n_contracts=7)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]


# ---------------------------------------------------------------------------
# SO3 (3M SONIA) strip
# ---------------------------------------------------------------------------
def test_fetch_sonia_so3_strip_columns_and_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"sonia_so3_strip_1y_to_{date_str}.parquet",
        root="SO3", n=8, val_date_str=date_str, base_price=96.00,
    )

    out = dcl.fetch_sonia_so3_strip(date_str, n_contracts=8)
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    assert len(out) == 8

    for _, r in out.iterrows():
        assert r["implied_rate"] == pytest.approx((100.0 - r["price"]) / 100.0)
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all()


def test_fetch_sonia_so3_strip_quarterly_step(tmp_path, monkeypatch):
    """SO3.c.0 on 2026-04-28 -> June quarter -> ref start 2026-06-17 (3rd Wed)."""
    import datetime as dt
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"sonia_so3_strip_1y_to_{date_str}.parquet",
        root="SO3", n=2, val_date_str=date_str,
    )

    out = dcl.fetch_sonia_so3_strip(date_str, n_contracts=8)
    val_date = dt.date.fromisoformat(date_str)
    expected_c0 = (dt.date(2026, 6, 17) - val_date).days / 365.25
    expected_c1 = (dt.date(2026, 9, 16) - val_date).days / 365.25  # 3rd Wed Sep 2026

    by_contract = dict(zip(out["contract"], out["expiry_years"]))
    assert by_contract["SO3.c.0"] == pytest.approx(expected_c0, abs=1e-6)
    assert by_contract["SO3.c.1"] == pytest.approx(expected_c1, abs=1e-6)


def test_fetch_sonia_so3_strip_missing_cache_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)
    out = dcl.fetch_sonia_so3_strip("2099-01-01", n_contracts=8)
    assert out.empty
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]


# ---------------------------------------------------------------------------
# SOA (1M SONIA) strip
# ---------------------------------------------------------------------------
def test_fetch_sonia_soa_strip_columns_and_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"sonia_soa_strip_1y_to_{date_str}.parquet",
        root="SOA", n=6, val_date_str=date_str, base_price=95.75,
    )

    out = dcl.fetch_sonia_soa_strip(date_str, n_contracts=6)
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    # SOA.c.0 falls on 2026-04-15 (3rd Wed Apr) which is < val_date, so 5 rows.
    assert 4 <= len(out) <= 6
    for _, r in out.iterrows():
        assert r["implied_rate"] == pytest.approx((100.0 - r["price"]) / 100.0)
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all()


def test_fetch_sonia_soa_strip_missing_cache_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)
    out = dcl.fetch_sonia_soa_strip("2099-01-01", n_contracts=6)
    assert out.empty


# ---------------------------------------------------------------------------
# FST3 (3M ESTR) strip
# ---------------------------------------------------------------------------
def test_fetch_estr_fst3_strip_columns_and_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"estr_fst3_strip_1y_to_{date_str}.parquet",
        root="FST3", n=8, val_date_str=date_str, base_price=97.50,
    )

    out = dcl.fetch_estr_fst3_strip(date_str, n_contracts=8)
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    assert len(out) == 8
    for _, r in out.iterrows():
        assert r["implied_rate"] == pytest.approx((100.0 - r["price"]) / 100.0)
    # First implied rate: (100 - 97.50) / 100 = 0.025
    assert out.iloc[0]["implied_rate"] == pytest.approx(0.025, abs=1e-9)

    diffs = out["expiry_years"].diff().dropna()
    assert (diffs >= 0).all()


def test_fetch_estr_fst3_strip_missing_cache_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)
    out = dcl.fetch_estr_fst3_strip("2099-01-01", n_contracts=8)
    assert out.empty


# ---------------------------------------------------------------------------
# FEMP (ECB-dated ESTR) strip
# ---------------------------------------------------------------------------
def test_fetch_estr_femp_strip_columns_and_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)

    date_str = "2026-04-28"
    _make_strip_cache(
        tmp_path, f"estr_femp_strip_1y_to_{date_str}.parquet",
        root="FEMP", n=6, val_date_str=date_str, base_price=97.80,
    )

    out = dcl.fetch_estr_femp_strip(date_str, n_contracts=6)
    assert list(out.columns) == ["contract", "expiry_years", "price", "implied_rate"]
    assert len(out) == 6
    for _, r in out.iterrows():
        assert r["implied_rate"] == pytest.approx((100.0 - r["price"]) / 100.0)
    # First implied rate: (100 - 97.80) / 100 = 0.022
    assert out.iloc[0]["implied_rate"] == pytest.approx(0.022, abs=1e-9)

    # Approximate ECB cycle: each step ~6 weeks (~42 days = 0.115Y).  The
    # increment between consecutive contracts should be roughly 42/365.25.
    diffs = out["expiry_years"].diff().dropna()
    assert (diffs > 0).all()
    expected_step = 42.0 / 365.25
    for d in diffs:
        assert d == pytest.approx(expected_step, abs=1e-6)


def test_fetch_estr_femp_strip_missing_cache_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dcl, "CACHE_DIR", tmp_path)
    out = dcl.fetch_estr_femp_strip("2099-01-01", n_contracts=6)
    assert out.empty
