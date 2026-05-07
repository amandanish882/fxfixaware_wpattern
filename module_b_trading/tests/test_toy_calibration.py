"""Tests for the turn-of-year calibration module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_calibrate_toy_returns_dict_with_g4_pairs(tmp_path):
    """With synthetic year-end widening in input, calibration recovers it."""
    from module_b_trading.toy_calibration import calibrate_toy_overlay

    rows = []
    for d in pd.date_range("2024-11-01", "2025-02-28", freq="B"):
        spot = 1.0850
        toy = (d.month == 12 and d.day >= 20)
        widening = 0.0050 if toy else 0.0020
        rows.append({"ts_event": d, "symbol": "6EH5.c.0", "close": spot})
        rows.append(
            {"ts_event": d, "symbol": "6EM5.c.1", "close": spot * (1.0 + widening)}
        )
    df = pd.DataFrame(rows)
    cache = tmp_path / "fx_curve_history.parquet"
    df.to_parquet(cache, index=False)

    out = calibrate_toy_overlay(cache_path=cache)
    assert "EUR/USD" in out
    # Synthetic widening was 50bp - 20bp = 30bp
    assert 25 <= out["EUR/USD"] <= 35


def test_calibrate_toy_handles_missing_cache(tmp_path):
    from module_b_trading.toy_calibration import calibrate_toy_overlay

    out = calibrate_toy_overlay(cache_path=tmp_path / "nonexistent.parquet")
    assert out == {}
