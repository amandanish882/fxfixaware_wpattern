"""
Tests for shared/foreign_ois_loader.py
======================================

Covers parsing of ECB / BoE / BoJ-FRED / RBA payloads and the public
fetch_foreign_ois_curve(...) entry point with HTTP patched out.

Run with:  pytest shared/tests/test_foreign_ois_loader.py -v
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup so imports resolve from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from shared import foreign_ois_loader as fol


# ---------------------------------------------------------------------------
# Fixtures: small fake payloads
# ---------------------------------------------------------------------------
FAKE_ECB_CSV = (
    "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y,B,U2,2026-04-28,2.456,A\n"
)

# BoE CSV: header + a couple of rows.  Last non-empty row is what the
# parser should use.
FAKE_BOE_CSV = (
    "DATE,IUDSOIA,IUDMNZC,IUDMNPC,IUDMNHC\n"
    "27 Apr 2026,4.20,4.05,4.10,4.30\n"
    "28 Apr 2026,4.25,4.10,4.15,4.35\n"
)

FAKE_FRED_JSON = json.dumps({
    "observations": [
        {"date": "2026-04-28", "value": "0.50"}
    ]
})

# RBA F1 CSV: a header block, blank lines, then the Series ID row, then dates.
FAKE_RBA_CSV = (
    "Title,Cash Rate Target,90-day Bank Bill,3y Govt,5y Govt,10y Govt\n"
    "Description,desc1,desc2,desc3,desc4,desc5\n"
    "Frequency,Daily,Daily,Daily,Daily,Daily\n"
    "Type,Original,Original,Original,Original,Original\n"
    "Units,Per cent,Per cent,Per cent,Per cent,Per cent\n"
    "Source,RBA,RBA,RBA,RBA,RBA\n"
    "Publication date,,,,,\n"
    "Series ID,FIRMMCRT,FIRMMBAB90,FCMYGBAG3,FCMYGBAG5,FCMYGBAG10\n"
    "27-Apr-2026,4.10,4.20,3.90,4.00,4.30\n"
    "28-Apr-2026,4.10,4.25,3.95,4.05,4.35\n"
)

# RBA F2 CSV: 2Y / 3Y / 5Y / 10Y CGS yields with the metadata header block.
FAKE_RBA_F2_CSV = (
    "Title,2y Govt,3y Govt,5y Govt,10y Govt\n"
    "Description,d1,d2,d3,d4\n"
    "Frequency,Daily,Daily,Daily,Daily\n"
    "Type,Original,Original,Original,Original\n"
    "Units,Per cent,Per cent,Per cent,Per cent\n"
    "Source,RBA,RBA,RBA,RBA\n"
    "Publication date,,,,\n"
    "Series ID,FCMYGBAG2D,FCMYGBAG3D,FCMYGBAG5D,FCMYGBAG10D\n"
    "27-Apr-2026,3.40,3.55,3.80,4.20\n"
    "28-Apr-2026,3.50,3.60,3.85,4.25\n"
)


# ---------------------------------------------------------------------------
# Parser-level tests (no HTTP, no mocking required)
# ---------------------------------------------------------------------------
def test_ecb_eur_ois_parses_csv():
    """_parse_ecb_csv -- last OBS_VALUE / 100 is returned under tenor 0.0."""
    result = fol._parse_ecb_csv(FAKE_ECB_CSV)
    assert 0.0 in result
    assert abs(result[0.0] - 0.02456) < 1e-9


def test_unsupported_currency_raises():
    """Any currency outside EUR/GBP/JPY/AUD must raise ValueError."""
    with pytest.raises(ValueError):
        fol.fetch_foreign_ois_curve("USD", "2026-04-28")


# ---------------------------------------------------------------------------
# End-to-end tests with _http_get patched
# ---------------------------------------------------------------------------
def test_boe_gbp_ois_returns_term_structure(tmp_path, monkeypatch):
    """Patched _http_get returns a fake BoE CSV; check tenor map."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    with patch.object(fol, "_http_get", return_value=FAKE_BOE_CSV):
        curve = fol.fetch_foreign_ois_curve("GBP", "2026-04-28")

    assert curve, "Expected a non-empty GBP curve"
    # Overnight (1/365)
    assert any(abs(t - 1.0 / 365.0) < 1e-9 for t in curve)
    # 1Y, 5Y, 10Y
    assert 1.0 in curve
    assert 5.0 in curve
    assert 10.0 in curve

    # Values from the last row of FAKE_BOE_CSV are 4.25 / 4.10 / 4.15 / 4.35
    assert abs(curve[1.0] - 0.0410) < 1e-9
    assert abs(curve[5.0] - 0.0415) < 1e-9
    assert abs(curve[10.0] - 0.0435) < 1e-9


def test_boj_jpy_ois_returns_at_least_overnight(tmp_path, monkeypatch):
    """JPY curve must have an overnight point from the FRED TONA-equivalent series."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    # Make sure get_fred_api_key returns something so the JPY path runs.
    monkeypatch.setattr(fol.config, "get_fred_api_key", lambda: "FAKE_KEY")
    with patch.object(fol, "_http_get", return_value=FAKE_FRED_JSON):
        curve = fol.fetch_foreign_ois_curve("JPY", "2026-04-28")

    assert curve, "Expected a non-empty JPY curve"
    overnight_tenor = 1.0 / 365.0
    assert any(abs(t - overnight_tenor) < 1e-9 for t in curve)

    # TONA = 0.50 / 100 = 0.005
    on_val = curve[overnight_tenor]
    assert abs(on_val - 0.005) < 1e-9


def test_jpy_loader_pulls_real_term_structure_from_fred(tmp_path, monkeypatch):
    """JPY loader Tier-2 should add 3M and 10Y from FRED on top of TONA overnight."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fol.config, "get_fred_api_key", lambda: "FAKE_KEY")
    fred_responses = {
        "IRSTCI01JPM156N": '{"observations":[{"date":"2026-04-24","value":"0.50"}]}',
        "IR3TIB01JPM156N": '{"observations":[{"date":"2026-04-24","value":"0.55"}]}',
        "IRLTLT01JPM156N": '{"observations":[{"date":"2026-04-24","value":"1.20"}]}',
    }

    def fake_get(url, headers=None, timeout=15):
        for sid, body in fred_responses.items():
            if sid in url:
                return body
        return ""

    with patch("shared.foreign_ois_loader._http_get", side_effect=fake_get):
        curve = fol.fetch_foreign_ois_curve("JPY", "2026-04-28")

    assert 0.25 in curve, f"missing 3M tenor in {curve}"
    assert 10.0 in curve
    assert curve[0.25] == pytest.approx(0.0055)
    assert curve[10.0] == pytest.approx(0.0120)


def test_aud_loader_includes_long_end_from_fred_when_rba_lacks_it(tmp_path, monkeypatch):
    """AUD Tier-1 (RBA F1) only gives short tenors; Tier-2 (FRED IRLTLT01AUM156N) fills 10Y."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fol.config, "get_fred_api_key", lambda: "FAKE_KEY")
    rba_csv = (
        "Series ID,FIRMMCRTD,FIRMMBAB90D\n"
        "28-Apr-2026,4.10,4.15\n"
    )
    fred_aud = '{"observations":[{"date":"2026-04-24","value":"4.40"}]}'

    def fake_get(url, headers=None, timeout=15):
        if "rba.gov.au" in url:
            return rba_csv
        if "IRLTLT01AUM156N" in url:
            return fred_aud
        return ""

    with patch("shared.foreign_ois_loader._http_get", side_effect=fake_get):
        curve = fol.fetch_foreign_ois_curve("AUD", "2026-04-28")

    assert 10.0 in curve
    assert curve[10.0] == pytest.approx(0.0440)
    # Tier-1 short tenors should still be present
    assert 0.25 in curve or (1.0 / 365.0) in curve


def test_rba_aud_ois_term_structure(tmp_path, monkeypatch):
    """RBA F1 + F2 CSVs parse cleanly into a multi-tenor AUD curve.

    Post-upgrade, the AUD loader should expose at least 5 tenors:
    overnight, 0.25, plus F2-derived bridge points (2Y, 3Y, 5Y) and a
    long end at 10Y.  The bridge points share their CGS yields with the
    spread-adjusted F2 path; with no FCMYGBAG2 in the synthetic F1 fixture
    and no separate F2 fixture, _aud_ois_cgs_spread degrades to 0.0 and
    bridge values pass through unchanged.
    """
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)

    def fake_get(url, headers=None, timeout=15):
        if "f2-data.csv" in url:
            return FAKE_RBA_F2_CSV
        if "f1-data.csv" in url:
            return FAKE_RBA_CSV
        return ""

    with patch.object(fol, "_http_get", side_effect=fake_get):
        curve = fol.fetch_foreign_ois_curve("AUD", "2026-04-28")

    assert curve, "Expected a non-empty AUD curve"
    # Post-upgrade target: at least 5 tenors among {ON, 0.25, 2, 3, 5, 10}.
    assert len(curve) >= 5, f"expected >=5 tenors, got {sorted(curve)}"

    # Spot-check a couple of values from the last row (28-Apr-2026):
    # FIRMMBAB90 = 4.25, FCMYGBAG10 = 4.35
    assert 0.25 in curve
    assert abs(curve[0.25] - 0.0425) < 1e-9
    assert 10.0 in curve
    assert abs(curve[10.0] - 0.0435) < 1e-9
    # F2-derived bridge: 2Y CGS = 3.50% from FAKE_RBA_F2_CSV
    assert 2.0 in curve
    # 3Y from F1 (3.95%) gets overwritten by F2-derived 3.60% (with 0 spread,
    # since FAKE_RBA_CSV's F1 has no FCMYGBAG2 column to anchor the spread).
    assert 3.0 in curve


def test_rba_f2_govt_yields_parses_csv(monkeypatch):
    """_rba_f2_govt_yields returns a tenor->yield map from F2 CSV."""
    with patch.object(fol, "_http_get", return_value=FAKE_RBA_F2_CSV):
        yields = fol._rba_f2_govt_yields("2026-04-28")

    # FAKE_RBA_F2_CSV last row: 2Y=3.50, 3Y=3.60, 5Y=3.85, 10Y=4.25
    assert len(yields) >= 3, f"expected >=3 F2 tenors, got {yields}"
    assert 2.0 in yields
    assert abs(yields[2.0] - 0.0350) < 1e-9
    assert 3.0 in yields
    assert abs(yields[3.0] - 0.0360) < 1e-9
    assert 10.0 in yields
    assert abs(yields[10.0] - 0.0425) < 1e-9


def test_rba_f2_govt_yields_returns_empty_on_failure(monkeypatch):
    """_rba_f2_govt_yields returns {} when the HTTP call raises."""
    def boom(*_a, **_k):
        raise RuntimeError("simulated network failure")
    with patch.object(fol, "_http_get", side_effect=boom):
        yields = fol._rba_f2_govt_yields("2026-04-28")
    assert yields == {}


def test_aud_ois_cgs_spread_basic(monkeypatch):
    """_aud_ois_cgs_spread = OIS_anchor - CGS_at_same_tenor (linearly interp).

    With the synthetic F1+F2 fixtures, cash=4.10%, 90D BAB=4.25%, 2Y CGS=3.50%.
    No F1 OIS columns are populated, so 90D BAB is the OIS anchor.  CGS at
    0.25y interp = 4.10 + (3.50-4.10) * (0.25/2) = 4.025%.  Spread =
    4.25 - 4.025 = +22.5 bp.
    """
    def fake_get(url, headers=None, timeout=15):
        if "f2-data.csv" in url:
            return FAKE_RBA_F2_CSV
        if "f1-data.csv" in url:
            return FAKE_RBA_CSV
        return ""

    with patch.object(fol, "_http_get", side_effect=fake_get):
        spread = fol._aud_ois_cgs_spread("2026-04-28")

    assert spread is not None, "spread should not be None when F1+F2 both available"
    # cash=0.041, bab=0.0425, cgs2y=0.035
    expected = 0.0425 - (0.041 + (0.035 - 0.041) * (0.25 / 2.0))
    assert abs(spread - expected) < 1e-9, f"got {spread}, expected {expected}"
    # Sanity: well within [-100bp, +50bp] band
    assert -0.010 < spread < 0.005


def test_aud_ois_cgs_spread_returns_none_when_f2_unavailable(monkeypatch):
    """_aud_ois_cgs_spread returns None if F2 fetch fails (or yields empty)."""
    def fake_get(url, headers=None, timeout=15):
        if "f2-data.csv" in url:
            return ""  # F2 unavailable
        if "f1-data.csv" in url:
            return FAKE_RBA_CSV
        return ""

    with patch.object(fol, "_http_get", side_effect=fake_get):
        spread = fol._aud_ois_cgs_spread("2026-04-28")
    assert spread is None


def test_aud_ois_cgs_spread_returns_none_when_f1_unavailable(monkeypatch):
    """_aud_ois_cgs_spread returns None if F1 (cash/BAB) is missing."""
    def fake_get(url, headers=None, timeout=15):
        if "f1-data.csv" in url:
            return ""  # F1 unavailable -- no cash rate, no BAB
        if "f2-data.csv" in url:
            return FAKE_RBA_F2_CSV
        return ""

    with patch.object(fol, "_http_get", side_effect=fake_get):
        spread = fol._aud_ois_cgs_spread("2026-04-28")
    assert spread is None


def test_cache_round_trip(tmp_path, monkeypatch):
    """Second call must hit the parquet cache, not _http_get."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)

    # First call: HTTP patched
    with patch.object(fol, "_http_get", return_value=FAKE_BOE_CSV) as mock_get:
        first = fol.fetch_foreign_ois_curve("GBP", "2026-04-28")
        assert first, "First call should return data"
        assert mock_get.call_count >= 1

    # Cache file must exist
    cache_file = tmp_path / "GBP_2026-04-28.parquet"
    assert cache_file.exists(), f"Expected cache at {cache_file}"

    # Second call: patch _http_get to RAISE if invoked, proving cache hit
    def _boom(*_a, **_k):
        raise AssertionError("_http_get should not be called -- cache miss!")

    with patch.object(fol, "_http_get", side_effect=_boom):
        second = fol.fetch_foreign_ois_curve("GBP", "2026-04-28")

    assert second == first, "Cached result must match the original fetch"


# ---------------------------------------------------------------------------
# Multi-tier fallback tests: when the primary source returns nothing,
# the loader must transparently fall back to FRED proxies.
# ---------------------------------------------------------------------------
def test_eur_falls_back_to_fred_when_ecb_blocked(tmp_path, monkeypatch):
    """When ECB returns nothing, EUR loader should query FRED EUR series."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fol.config, "get_fred_api_key", lambda: "FAKE_KEY")
    fred_eur_json = '{"observations":[{"date":"2026-04-24","value":"3.10"}]}'

    def fake_http_get(url, headers=None, timeout=15):
        if "ecb.europa.eu" in url:
            return ""  # ECB returns empty
        if "stlouisfed.org" in url and "EZM156N" in url:
            return fred_eur_json
        return ""

    with patch("shared.foreign_ois_loader._http_get", side_effect=fake_http_get):
        curve = fol.fetch_foreign_ois_curve("EUR", "2026-04-28")

    # Should have at least 1 tenor from FRED proxy
    assert len(curve) >= 1
    assert any(0.025 <= rate <= 0.035 for rate in curve.values())


def test_gbp_falls_back_to_fred_when_boe_blocked(tmp_path, monkeypatch):
    """When BoE IADB returns nothing, GBP loader should query FRED GBP series."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fol.config, "get_fred_api_key", lambda: "FAKE_KEY")
    fred_gbp_json = '{"observations":[{"date":"2026-04-24","value":"4.20"}]}'

    def fake_http_get(url, headers=None, timeout=15):
        if "bankofengland.co.uk" in url:
            return ""  # BoE returns empty
        if "stlouisfed.org" in url and "GBM156N" in url:
            return fred_gbp_json
        return ""

    with patch("shared.foreign_ois_loader._http_get", side_effect=fake_http_get):
        curve = fol.fetch_foreign_ois_curve("GBP", "2026-04-28")

    assert len(curve) >= 1
    assert any(0.030 <= rate <= 0.050 for rate in curve.values())


def test_aud_falls_back_to_fred_when_rba_blocked(tmp_path, monkeypatch):
    """When RBA returns nothing, AUD loader should query FRED AUD series."""
    monkeypatch.setattr(fol, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fol.config, "get_fred_api_key", lambda: "FAKE_KEY")
    fred_aud_json = '{"observations":[{"date":"2026-04-24","value":"4.30"}]}'

    def fake_http_get(url, headers=None, timeout=15):
        if "rba.gov.au" in url:
            return ""
        if "stlouisfed.org" in url and "AUM156N" in url:
            return fred_aud_json
        return ""

    with patch("shared.foreign_ois_loader._http_get", side_effect=fake_http_get):
        curve = fol.fetch_foreign_ois_curve("AUD", "2026-04-28")

    assert len(curve) >= 1
    assert any(0.030 <= rate <= 0.050 for rate in curve.values())
