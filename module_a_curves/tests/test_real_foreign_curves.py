"""
Tests for FXDataLoader.get_real_foreign_ois_curve and get_real_usd_ois_curve
============================================================================

Verifies that the new bridge methods on ``FXDataLoader`` correctly delegate
to the free-API foreign OIS loader and the Databento SOFR strip loader, and
gracefully handle missing/empty data.

Run with:  pytest module_a_curves/tests/test_real_foreign_curves.py -v
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so imports resolve from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unittest.mock import patch

import pandas as pd
import pytest

from module_a_curves.data_loader import FXDataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def loader():
    """Plain FXDataLoader with no FRED key (we don't hit live APIs)."""
    return FXDataLoader(fred_api_key=None)


@pytest.fixture
def synthetic_sofr_strip():
    """4-row SR3 strip with gently downward-sloping forwards."""
    return pd.DataFrame(
        {
            "contract": ["SR3M6", "SR3U6", "SR3Z6", "SR3H7"],
            "expiry_years": [0.13, 0.38, 0.63, 0.88],
            "price": [96.50, 96.60, 96.70, 96.85],
            "implied_rate": [0.0350, 0.0340, 0.0330, 0.0315],
        }
    )


# ---------------------------------------------------------------------------
# Tests -- get_real_foreign_ois_curve
# ---------------------------------------------------------------------------
def test_get_real_foreign_ois_curve_eur_returns_term_structure(loader):
    """Delegates to fetch_foreign_ois_curve and returns the dict unchanged."""
    fake_curve = {0.25: 0.024, 1.0: 0.025, 5.0: 0.027, 10.0: 0.029}
    with patch(
        "module_a_curves.data_loader.fetch_foreign_ois_curve",
        return_value=fake_curve,
    ) as mock_fetch:
        result = loader.get_real_foreign_ois_curve("EUR", "2026-04-28")

    assert result == fake_curve
    mock_fetch.assert_called_once_with("EUR", "2026-04-28")


def test_get_real_foreign_ois_curve_returns_empty_on_failure(loader):
    """When the source returns {}, the method passes through an empty dict."""
    with patch(
        "module_a_curves.data_loader.fetch_foreign_ois_curve",
        return_value={},
    ):
        result = loader.get_real_foreign_ois_curve("EUR", "2026-04-28")

    assert result == {}


# ---------------------------------------------------------------------------
# Tests -- get_real_usd_ois_curve
# ---------------------------------------------------------------------------
def test_get_real_usd_ois_curve_returns_curve_when_strip_present(
    loader, synthetic_sofr_strip
):
    """A non-empty strip should yield a usable DiscountCurve."""
    with patch(
        "module_a_curves.data_loader.fetch_sofr_strip",
        return_value=synthetic_sofr_strip,
    ):
        curve = loader.get_real_usd_ois_curve("2026-04-28")

    assert curve is not None
    assert hasattr(curve, "df")
    assert hasattr(curve, "zero_rate")
    # Sanity: discount factor at 6M should be < 1.0
    assert curve.df(0.5) < 1.0


def test_get_real_usd_ois_curve_returns_none_on_empty_strip(loader):
    """An empty strip DataFrame should produce None, not an exception."""
    empty = pd.DataFrame(
        columns=["contract", "expiry_years", "price", "implied_rate"]
    )
    with patch(
        "module_a_curves.data_loader.fetch_sofr_strip",
        return_value=empty,
    ):
        result = loader.get_real_usd_ois_curve("2026-04-28")

    assert result is None
