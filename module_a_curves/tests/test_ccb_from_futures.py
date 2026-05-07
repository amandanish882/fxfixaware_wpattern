"""
Tests for `CrossCurrencyBasis.from_cme_futures` classmethod.

Covers:
- CIP-perfect futures recover ~zero basis.
- Futures priced with a 50 bp synthetic basis recover ~50 bps back.
- Default constructor still works and preserves the existing schema.

Run with:  pytest module_a_curves/tests/test_ccb_from_futures.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from module_a_curves.fx_forward_curve import CrossCurrencyBasis


# ---------------------------------------------------------------------------
# Test stub: a flat-rate discount curve, df(t) = exp(-r * t)
# ---------------------------------------------------------------------------
class _FlatCurve:
    def __init__(self, rate):
        self.rate = rate

    def df(self, t):
        return float(np.exp(-self.rate * t))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def spot():
    return 1.0832  # EUR/USD


@pytest.fixture
def usd():
    return _FlatCurve(0.0430)  # USD ~ 4.30%


@pytest.fixture
def eur():
    return _FlatCurve(0.0250)  # EUR ~ 2.50%


@pytest.fixture
def tenors():
    # Match the standard quarterly / annual CME schedule.
    return [
        ("EURH26", 0.25),
        ("EURM26", 0.50),
        ("EURU26", 0.75),
        ("EURZ26", 1.00),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_ccb_from_futures_recovers_zero_when_cip_holds(spot, usd, eur, tenors):
    """If futures prices equal CIP forwards exactly, recovered basis ~ 0.

    Project convention (FOREIGN/USD with S = USD per FOREIGN):
        F_cip = S * df_foreign / df_usd
    """
    rows = []
    for contract, t in tenors:
        # Correct CIP forward in project's "FOREIGN/USD" convention.
        f_cip = spot * eur.df(t) / usd.df(t)
        rows.append({"contract": contract, "expiry_years": t, "price": f_cip})
    futures_df = pd.DataFrame(rows)

    ccb = CrossCurrencyBasis.from_cme_futures(
        spot=spot,
        usd_curve=usd,
        foreign_curve=eur,
        futures_df=futures_df,
        pair="EUR/USD",
    )
    curve = ccb.basis_curve("EUR/USD")

    assert not curve.empty
    assert abs(curve["basis_bps"]).max() < 1.0, (
        f"Expected ~0 basis, got max |bps| = {abs(curve['basis_bps']).max():.3f}"
    )


def test_ccb_from_futures_recovers_known_50bp_basis(spot, usd, eur, tenors):
    """A market basis of -50 bps means F_obs = F_cip * exp(+50bp * t).

    The simple-rate equivalent: bumping the foreign rate UP by 50 bp makes the
    foreign side compound faster, lowering df_foreign(t) — which under the
    correct F_cip = S * df_for / df_usd formula reduces F_obs *below* the
    no-basis CIP forward, exactly the negative-basis sign convention.

    We construct futures using a foreign curve with rates raised by 50 bp,
    pass the original (unbumped) curve into the inversion, and expect a
    recovered basis of approximately -50 bps.
    """
    eur_adj = _FlatCurve(0.0250 + 0.005)  # foreign side bumped 50 bps higher

    rows = []
    for contract, t in tenors:
        # Build observed futures using the bumped foreign curve (CORRECT CIP convention).
        f_obs = spot * eur_adj.df(t) / usd.df(t)
        rows.append({"contract": contract, "expiry_years": t, "price": f_obs})
    futures_df = pd.DataFrame(rows)

    # Pass the ORIGINAL eur (without 50 bp bump) to the inversion.
    ccb = CrossCurrencyBasis.from_cme_futures(
        spot=spot,
        usd_curve=usd,
        foreign_curve=eur,
        futures_df=futures_df,
        pair="EUR/USD",
    )
    curve = ccb.basis_curve("EUR/USD")

    assert not curve.empty
    # f_obs = S * (df_eur_adj/df_usd) and f_cip = S * (df_eur/df_usd), so
    # f_cip/f_obs = df_eur / df_eur_adj = exp((r_adj - r_eur) * t) = exp(+50bp * t).
    # ccb = -ln(f_cip/f_obs)/t = -50 bp/yr. Recovered basis should be ~ -50 bp.
    for _, row in curve.iterrows():
        assert -55.0 < row["basis_bps"] < -45.0, (
            f"At tenor {row['tenor']}: expected ~-50 bps, got {row['basis_bps']:.3f}"
        )


def test_default_ccb_unchanged():
    """Default constructor still returns the legacy hardcoded curve schema."""
    ccb = CrossCurrencyBasis()
    df = ccb.basis_curve("EUR/USD")

    assert not df.empty
    # Schema must match the existing schema used by the rest of the project.
    assert list(df.columns) == ["tenor", "maturity_years", "basis_bps"]
    # Sanity: legacy EUR/USD 1Y basis is -21 bps in the hardcoded table.
    one_y = df[df["tenor"] == "1Y"]
    assert not one_y.empty
    assert abs(one_y["basis_bps"].iloc[0] - (-21.0)) < 1e-9
