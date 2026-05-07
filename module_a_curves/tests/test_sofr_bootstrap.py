"""
Tests for SOFR Futures Bootstrap (Module A)
============================================

Covers: monotone-decreasing discount factors, plausible zero-rate band,
empty-strip error handling, and the optional overnight anchor rate.

Run with:  pytest module_a_curves/tests/test_sofr_bootstrap.py -v
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so imports resolve from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from module_a_curves.sofr_futures_bootstrap import bootstrap_usd_ois_from_strip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_strip():
    """Realistic 5-contract SR3 strip with gently downward-sloping forwards."""
    return pd.DataFrame(
        {
            "contract": ["SR3M6", "SR3U6", "SR3Z6", "SR3H7", "SR3M7"],
            "expiry_years": [0.13, 0.38, 0.63, 0.88, 1.13],
            "implied_rate": [0.0350, 0.0340, 0.0330, 0.0315, 0.0300],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_bootstrap_returns_monotone_decreasing_discount_factors(synthetic_strip):
    """df should decrease monotonically as t increases (positive rates)."""
    curve = bootstrap_usd_ois_from_strip(
        strip=synthetic_strip,
        valuation_date="2026-04-28",
    )
    d_025 = curve.df(0.25)
    d_050 = curve.df(0.50)
    d_100 = curve.df(1.00)
    d_150 = curve.df(1.50)
    assert d_025 > d_050 > d_100 > d_150
    # Sanity: all strictly less than 1.0
    assert d_025 < 1.0


def test_bootstrap_zero_rate_in_plausible_band(synthetic_strip):
    """1Y zero rate should lie within the band of supplied implied rates."""
    curve = bootstrap_usd_ois_from_strip(
        strip=synthetic_strip,
        valuation_date="2026-04-28",
    )
    z_1y = curve.zero_rate(1.0)
    assert 0.025 <= z_1y <= 0.045


def test_bootstrap_empty_strip_raises():
    """Empty strip with the right columns should raise ValueError."""
    empty = pd.DataFrame(columns=["contract", "expiry_years", "implied_rate"])
    with pytest.raises(ValueError, match="Empty SOFR strip"):
        bootstrap_usd_ois_from_strip(
            strip=empty,
            valuation_date="2026-04-28",
        )


def test_bootstrap_uses_provided_overnight_anchor(synthetic_strip):
    """When overnight_rate is supplied, df(1/365) should match continuous
    compounding of that rate.  SOFR is a daily-compounded rate, treated as
    continuous over short stubs to avoid simple-vs-continuous convention
    drops between t=1/365 and the first SR3 IMM."""
    import math
    on_rate = 0.04
    curve = bootstrap_usd_ois_from_strip(
        strip=synthetic_strip,
        valuation_date="2026-04-28",
        overnight_rate=on_rate,
    )
    expected = math.exp(-on_rate / 365.0)
    assert curve.df(1.0 / 365.0) == pytest.approx(expected, abs=1e-9)


def test_bootstrap_expiry_years_is_imm_start_of_3m_window():
    """``expiry_years`` is the IMM date; the SR3 forward applies over
    ``[expiry_years, expiry_years + 0.25]``.

    With a flat 3.66% strip and a 3.64% overnight anchor, the curve should
    pin the front to ~3.64% (the FRED SOFR fixing) and rise smoothly toward
    the SR3 forward -- no dip below the anchor at the first IMM node.
    """
    strip = pd.DataFrame({
        "contract":     ["SR3M6", "SR3U6", "SR3Z6", "SR3H7"],
        # IMM dates from val_date = 2026-04-28:
        "expiry_years": [0.137, 0.386, 0.635, 0.884],
        "implied_rate": [0.03665, 0.03660, 0.03665, 0.03670],
    })
    on_rate = 0.0364
    curve = bootstrap_usd_ois_from_strip(
        strip=strip,
        valuation_date="2026-04-28",
        overnight_rate=on_rate,
    )
    # No dip: zero rate at the first IMM date and at ref_end of the first
    # contract should not fall below the overnight anchor.
    z_imm = curve.zero_rate(0.137)
    z_ref_end = curve.zero_rate(0.387)
    assert z_imm >= on_rate - 1e-4, f"front-stub dip: z(0.137)={z_imm:.5f}"
    assert z_ref_end >= on_rate - 1e-4, f"front dip: z(0.387)={z_ref_end:.5f}"
    # And both should sit between the overnight anchor and the SR3 forward.
    assert on_rate - 1e-4 <= z_ref_end <= 0.0367


def test_bootstrap_curve_nodes_are_at_imm_and_ref_end():
    """Each contract should add a node at ref_start and at ref_end (= ref_start + 0.25)."""
    strip = pd.DataFrame({
        "contract":     ["SR3M6", "SR3U6"],
        "expiry_years": [0.137, 0.386],
        "implied_rate": [0.03665, 0.03660],
    })
    curve = bootstrap_usd_ois_from_strip(
        strip=strip,
        valuation_date="2026-04-28",
        overnight_rate=0.0364,
    )
    times = list(curve.times)
    # Expected node layout: 0, 1/365, 0.137, 0.387, 0.386 (skipped, behind), 0.636
    # (Second contract's ref_start 0.386 < prev_t 0.387, so the overlap path
    # produces a node at 0.636 = 0.386 + 0.25.)
    assert times[0] == 0.0
    assert times[1] == pytest.approx(1 / 365.0)
    assert any(abs(t - 0.137) < 1e-6 for t in times), times
    assert any(abs(t - 0.387) < 1e-6 for t in times), times
    assert any(abs(t - 0.636) < 1e-6 for t in times), times
