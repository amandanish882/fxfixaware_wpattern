"""Tests for FX swap mechanics: forward points, T/N, turn-of-period overlay."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from module_b_trading.fx_swap_mechanics import (
    forward_points,
    pip_factor,
)


def test_forward_points_usdjpy_3m_negative_159_pips():
    """Worked example: USD/JPY=150, USD 4.5%, JPY 0.25%, 3M -> ~-159 pips, JPY at premium."""
    fwd, points_in_pips = forward_points(
        spot=150.00,
        rate_base=0.045,    # USD funding rate (USD is base in USD/JPY)
        rate_quote=0.0025,  # JPY funding rate
        tenor_years=0.25,
        pair="USD/JPY",
    )
    # F = 150 * (1 + 0.0025*0.25) / (1 + 0.045*0.25) ~ 148.42
    assert abs(fwd - 148.42) < 0.05, f"forward={fwd}"
    # Pip factor 100 (since USD/JPY is quoted to 2 decimals); F-S = -1.58 -> -158 pips
    assert -160 <= points_in_pips <= -155, f"points={points_in_pips}"


def test_forward_points_eurusd_1y_positive_when_eur_lower():
    """EUR/USD=1.08, EUR 2.5%, USD 4.5%, 1Y -> positive forward points (EUR at premium)."""
    fwd, points = forward_points(
        spot=1.0800,
        rate_base=0.025,
        rate_quote=0.045,
        tenor_years=1.0,
        pair="EUR/USD",
    )
    # F ~ 1.08 * 1.045 / 1.025 = 1.1011 -> +211 pips
    assert 1.099 < fwd < 1.103
    assert 200 <= points <= 220


def test_pip_factor_jpy_pair_is_100():
    assert pip_factor("USD/JPY") == 100
    assert pip_factor("JPY/USD") == 100


def test_pip_factor_majors_are_10000():
    assert pip_factor("EUR/USD") == 10000
    assert pip_factor("GBP/USD") == 10000
    assert pip_factor("AUD/USD") == 10000


def test_tn_roll_one_business_day_returns_small_negative_points_for_jpy():
    """T/N for USD/JPY: roll spot from T+2 to T+1, equivalent to a 1-day swap."""
    from module_b_trading.fx_swap_mechanics import tn_roll_points

    pts = tn_roll_points(
        spot=150.00,
        rate_base=0.045,
        rate_quote=0.0025,
        pair="USD/JPY",
    )
    # 1-day T/N for USD/JPY (rate diff 4.25% over 1/365 yr):
    # F - S = 150 * ((1+0.0025/365)/(1+0.045/365) - 1) ~ 150 * (-1.164e-4)
    # = -0.01746 yen, times pip factor 100 = -1.746 pips. Small and signed correctly.
    assert -2.0 < pts < 0


def test_turn_overlay_widens_short_dated_points_into_year_end():
    """1W swap points widen (in absolute value) when the period crosses Dec 31."""
    from module_b_trading.fx_swap_mechanics import (
        forward_points,
        apply_turn_overlay,
    )

    base_pts = forward_points(150.00, 0.045, 0.0025, 7/365, "USD/JPY")[1]
    # Period: 2026-12-29 -> 2026-01-05 crosses year-end
    overlay = apply_turn_overlay(
        base_points=base_pts,
        start_date="2026-12-29",
        end_date="2027-01-05",
        pair="USD/JPY",
    )
    # Year-end overlay should make the absolute value LARGER (more negative for USD/JPY)
    assert abs(overlay) > abs(base_pts)
    assert overlay < base_pts  # more negative
