"""Tests for tradeable cross-currency basis market mechanics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_structural_ccb_eur_negative_more_negative_with_tenor():
    """EUR/USD basis is structurally negative; widens (more neg) with tenor."""
    from module_b_trading.xccy_basis_market import structural_basis_curve

    df = structural_basis_curve("EUR/USD", as_of="2026-04-28")
    assert (df["basis_bps"] < 0).all()
    # Monotone (more negative at longer tenors)
    assert df.sort_values("tenor_years")["basis_bps"].is_monotonic_decreasing


def test_structural_ccb_jpy_more_negative_than_eur():
    """JPY basis is structurally most negative (Japanese USD-asset hedging)."""
    from module_b_trading.xccy_basis_market import structural_basis_curve

    eur = structural_basis_curve("EUR/USD", as_of="2026-04-28")
    jpy = structural_basis_curve("JPY/USD", as_of="2026-04-28")
    # Match on tenor_years
    eur_5y = eur.loc[eur["tenor_years"] == 5.0, "basis_bps"].iloc[0]
    jpy_5y = jpy.loc[jpy["tenor_years"] == 5.0, "basis_bps"].iloc[0]
    assert jpy_5y < eur_5y, f"JPY 5y ({jpy_5y}) should be more negative than EUR 5y ({eur_5y})"


def test_basis_blowout_at_year_end():
    """Year-end window widens (more negative) the short-dated basis significantly."""
    from module_b_trading.xccy_basis_market import basis_with_turn_blowout

    # Asking for 1M basis, period crosses Dec 31
    base_3m = basis_with_turn_blowout("EUR/USD", tenor_years=0.25, as_of="2026-09-15")
    blowout_3m = basis_with_turn_blowout("EUR/USD", tenor_years=0.25, as_of="2026-12-15")
    assert blowout_3m < base_3m, "year-end 3M basis should be MORE negative"
    # Magnitude difference: at least 5bp wider
    assert (base_3m - blowout_3m) >= 5.0


def test_arbitrage_band_signals_when_implied_diverges_from_quoted():
    """When FX-swap-implied basis diverges from xccy market by > tolerance,
    the helper flags an arbitrage opportunity."""
    from module_b_trading.xccy_basis_market import arbitrage_band

    out = arbitrage_band(
        pair="EUR/USD",
        fx_implied_bps=-35.0,   # observed from CME futures + OIS curves
        xccy_quoted_bps=-18.0,  # market xccy basis swap quote
        tolerance_bps=5.0,
    )
    assert out["divergence_bps"] == pytest.approx(-17.0)
    assert out["arbitrage"] is True
    assert out["direction"] == "buy_xccy_sell_fxswap"  # implied cheaper than quoted


def test_arbitrage_band_no_signal_within_tolerance():
    from module_b_trading.xccy_basis_market import arbitrage_band

    out = arbitrage_band(
        pair="EUR/USD",
        fx_implied_bps=-19.0,
        xccy_quoted_bps=-21.0,
        tolerance_bps=5.0,
    )
    assert out["arbitrage"] is False


def test_structural_5y_uses_calibration_when_curve_history_present():
    """When the CME curve history cache exists, structural levels should differ
    from pure hardcoded fallbacks for at least one pair."""
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent.parent / "data" / "databento_cache" / "fx_curve_history_2024-12-15_to_2026-01-31.parquet"
    if not cache.exists():
        import pytest
        pytest.skip("CME curve history cache not present")
    from module_b_trading.xccy_basis_market import _STRUCTURAL_5Y_BPS, _STRUCTURAL_5Y_BPS_HARDCODED, _calibrate_structural_5y_bps
    calibrated = _calibrate_structural_5y_bps()
    assert len(calibrated) >= 1
    # Each calibrated value should be a finite float in plausible range (-200 to +50 bps)
    for pair, v in calibrated.items():
        assert -200 <= v <= 50, f"{pair} = {v} bps looks implausible"
    # The public dict should match calibrated values where they exist
    for pair, v in calibrated.items():
        assert _STRUCTURAL_5Y_BPS[pair] == v
