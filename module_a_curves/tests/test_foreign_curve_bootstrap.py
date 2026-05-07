"""
Tests for module_a_curves.foreign_curve_bootstrap
=================================================

Covers the generic per-segment futures-strip bootstrap:
* empty strip returns a graceful one-node stub (does not raise),
* non-empty 3M strip produces monotone-decreasing discount factors,
* non-empty 1M strip (ref_period_yr = 1/12) also works,
* the supplied overnight anchor pins df(1/365) to exp(-r/365).

Run with:  pytest module_a_curves/tests/test_foreign_curve_bootstrap.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math

import pandas as pd
import pytest

from module_a_curves.foreign_curve_bootstrap import bootstrap_foreign_ois_from_strip


@pytest.fixture
def synthetic_3m_strip():
    """4-row 3M-RFR strip (SO3-style) with gently rising forwards."""
    return pd.DataFrame(
        {
            "contract": ["SO3.c.0", "SO3.c.1", "SO3.c.2", "SO3.c.3"],
            "expiry_years": [0.137, 0.386, 0.635, 0.884],
            "implied_rate": [0.0374, 0.0376, 0.0378, 0.0380],
        }
    )


@pytest.fixture
def synthetic_1m_strip():
    """6-row 1M-RFR strip (SR1-style)."""
    return pd.DataFrame(
        {
            "contract": [f"SR1.c.{i}" for i in range(1, 7)],
            "expiry_years": [0.06, 0.14, 0.21, 0.31, 0.39, 0.48],
            "implied_rate": [0.0364, 0.0365, 0.0364, 0.0363, 0.0364, 0.0363],
        }
    )


def test_empty_strip_returns_stub_curve_without_raising():
    """Empty strip should return a graceful one-node curve."""
    empty = pd.DataFrame(columns=["contract", "expiry_years", "implied_rate"])
    curve = bootstrap_foreign_ois_from_strip(
        strip=empty, valuation_date="2026-04-28", overnight_rate=0.04,
    )
    assert curve is not None
    # Two nodes: t=0 anchor + t=1/365 (degenerate flat-at-1.0 stub).
    assert len(curve.times) <= 2


def test_3m_strip_produces_monotone_decreasing_dfs(synthetic_3m_strip):
    curve = bootstrap_foreign_ois_from_strip(
        strip=synthetic_3m_strip,
        valuation_date="2026-04-28",
        overnight_rate=0.0373,
        ref_period_yr=0.25,
    )
    d_025 = curve.df(0.25)
    d_050 = curve.df(0.50)
    d_100 = curve.df(1.00)
    assert d_025 > d_050 > d_100
    assert d_025 < 1.0


def test_1m_strip_produces_monotone_decreasing_dfs(synthetic_1m_strip):
    curve = bootstrap_foreign_ois_from_strip(
        strip=synthetic_1m_strip,
        valuation_date="2026-04-28",
        overnight_rate=0.0363,
        ref_period_yr=1.0 / 12.0,
    )
    d_010 = curve.df(0.10)
    d_030 = curve.df(0.30)
    d_050 = curve.df(0.50)
    assert d_010 > d_030 > d_050


def test_overnight_anchor_pins_df_at_1_over_365(synthetic_3m_strip):
    """df(1/365) should be exp(-overnight_rate / 365) (continuous compounding)."""
    on_rate = 0.025
    curve = bootstrap_foreign_ois_from_strip(
        strip=synthetic_3m_strip,
        valuation_date="2026-04-28",
        overnight_rate=on_rate,
        ref_period_yr=0.25,
    )
    expected = math.exp(-on_rate / 365.0)
    assert curve.df(1.0 / 365.0) == pytest.approx(expected, abs=1e-9)
