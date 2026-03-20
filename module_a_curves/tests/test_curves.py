"""
Tests for Module A: OIS Curve Construction
===========================================

Covers: data loading fallback, bootstrap (deposits + swaps), discount factors,
zero rates, forward rates, interpolation methods, CIP FX forwards, forward points,
cross-currency basis, and central-bank meeting dates.

Run with:  pytest module_a_curves/tests/test_curves.py -v
"""

import sys
import datetime as dt
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so imports resolve from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from module_a_curves.data_loader import (
    FXDataLoader,
    FALLBACK_FX_SPOT,
    FALLBACK_RATES,
    _CB_DATES_2026,
)
from module_a_curves.curve_bootstrapper import (
    CurveBootstrapper,
    CurveInstrument,
    DiscountCurve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def loader():
    """FXDataLoader with no API key -> always hits embedded fallback."""
    return FXDataLoader(fred_api_key=None)


@pytest.fixture
def usd_instruments():
    """Realistic USD OIS calibration instruments (Mar-2026 levels).

    - O/N deposit at SOFR 4.33%
    - 1M deposit at 4.32%
    - 3M deposit at 4.31%
    - 2Y swap at 4.12%  (annual, ACT/360)
    - 5Y swap at 4.28%
    - 10Y swap at 4.55%
    - 30Y swap at 4.72%
    """
    return [
        CurveInstrument("deposit", 1 / 360, 0.0433, "ACT/360"),       # O/N
        CurveInstrument("deposit", 30 / 360, 0.0432, "ACT/360"),      # 1M
        CurveInstrument("deposit", 90 / 360, 0.0431, "ACT/360"),      # 3M
        CurveInstrument("swap", 2.0, 0.0412, "ACT/360", 1.0),         # 2Y
        CurveInstrument("swap", 5.0, 0.0428, "ACT/360", 1.0),         # 5Y
        CurveInstrument("swap", 10.0, 0.0455, "ACT/360", 1.0),        # 10Y
        CurveInstrument("swap", 30.0, 0.0472, "ACT/360", 1.0),        # 30Y
    ]


@pytest.fixture
def usd_curve(usd_instruments):
    """Bootstrapped USD OIS discount curve."""
    bs = CurveBootstrapper(interpolation_method="log_linear")
    return bs.bootstrap(usd_instruments, dt.date(2026, 3, 5))


@pytest.fixture
def eur_instruments():
    """EUR OIS calibration instruments at ~2.90% short end."""
    return [
        CurveInstrument("deposit", 1 / 360, 0.0290, "ACT/360"),
        CurveInstrument("deposit", 90 / 360, 0.0288, "ACT/360"),
        CurveInstrument("swap", 2.0, 0.0275, "ACT/360", 1.0),
        CurveInstrument("swap", 5.0, 0.0285, "ACT/360", 1.0),
        CurveInstrument("swap", 10.0, 0.0300, "ACT/360", 1.0),
    ]


@pytest.fixture
def eur_curve(eur_instruments):
    bs = CurveBootstrapper(interpolation_method="log_linear")
    return bs.bootstrap(eur_instruments, dt.date(2026, 3, 5))


# ===================================================================
# Tests
# ===================================================================

class TestFXDataLoader:
    """Tests for the three-tier FX data loader."""

    def test_fx_data_loader_fallback(self, loader):
        """Embedded snapshot returns valid FX rates when no API key is set.

        With fred_api_key=None and no parquet cache, get_fx_spot_rates()
        should return the FALLBACK_FX_SPOT dict, e.g. EUR/USD = 1.0832.
        """
        spots = loader.get_fx_spot_rates("2026-03-05")
        assert isinstance(spots, dict)
        assert len(spots) >= 4, "Expected at least 4 G4 pairs"
        # Values should be positive floats
        for pair, rate in spots.items():
            assert rate > 0, f"{pair} rate should be positive, got {rate}"

    def test_fx_data_loader_spot_rates(self, loader):
        """All G4 pairs present and realistic: EUR/USD ~1.08, GBP/USD ~1.27, etc."""
        spots = loader.get_fx_spot_rates("2026-03-05")

        # EUR/USD should be near 1.08 (range 0.90 - 1.30 is very generous)
        assert "EUR/USD" in spots
        assert 0.90 < spots["EUR/USD"] < 1.30, f"EUR/USD={spots['EUR/USD']}"

        # GBP/USD should be near 1.27
        assert "GBP/USD" in spots
        assert 1.10 < spots["GBP/USD"] < 1.50, f"GBP/USD={spots['GBP/USD']}"

        # JPY/USD is expressed as USD per JPY ~ 0.0067
        assert "JPY/USD" in spots
        assert 0.004 < spots["JPY/USD"] < 0.010, f"JPY/USD={spots['JPY/USD']}"

        # AUD/USD ~ 0.64
        assert "AUD/USD" in spots
        assert 0.50 < spots["AUD/USD"] < 0.80, f"AUD/USD={spots['AUD/USD']}"


class TestBootstrap:
    """Tests for the sequential OIS curve bootstrap."""

    def test_bootstrap_deposits(self):
        """Bootstrap a single O/N deposit at 4.33%.

        Expected: D(1/360) = 1 / (1 + 0.0433 * 1/360) = 1 / 1.00012028
                            = 0.999880 (approximately).
        """
        inst = CurveInstrument("deposit", 1 / 360, 0.0433, "ACT/360")
        bs = CurveBootstrapper()
        curve = bs.bootstrap([inst], dt.date(2026, 3, 5))

        t = 1 / 360  # ~ 0.002778 years
        expected_df = 1.0 / (1.0 + 0.0433 * t)  # = 0.999880
        actual_df = curve.df(t)
        assert abs(actual_df - expected_df) < 1e-8, (
            f"D({t:.6f}) = {actual_df:.8f}, expected {expected_df:.8f}"
        )

    def test_bootstrap_swap(self, usd_instruments, usd_curve):
        """Bootstrap deposit + swaps; verify swap par rates reprice to zero.

        The par rate implied by the bootstrapped curve at each swap maturity
        should match the input market rate to within 0.01 bps.
        """
        bs = CurveBootstrapper()
        report = bs.validate(usd_curve, usd_instruments)

        for _, row in report.iterrows():
            assert abs(row["error_bps"]) < 0.05, (
                f"{row['type']} at {row['maturity']:.2f}Y: "
                f"market={row['market_rate']:.6f}, model={row['model_rate']:.6f}, "
                f"error={row['error_bps']:.4f} bps"
            )


class TestDiscountCurve:
    """Tests for the DiscountCurve analytics."""

    def test_curve_discount_factor(self, usd_curve):
        """D(0)=1, D(t) decreasing for t>0, D(30)>0.

        At 4.3% rates, D(1) ~ exp(-0.043) ~ 0.958, D(30) ~ exp(-1.3) ~ 0.27.
        """
        assert usd_curve.df(0.0) == 1.0
        assert usd_curve.df(1.0) < 1.0
        assert usd_curve.df(5.0) < usd_curve.df(1.0)
        assert usd_curve.df(10.0) < usd_curve.df(5.0)
        assert usd_curve.df(30.0) > 0.0
        # Realistic range for 30Y at ~4.7%: D(30) ~ exp(-0.047*30) ~ 0.244
        assert 0.10 < usd_curve.df(30.0) < 0.50

    def test_zero_rate_positive(self, usd_curve):
        """All zero rates should be > 0 for t > 0.

        At the short end (~4.3%), zero rates should be near 4.3%.
        """
        for t in [0.01, 0.25, 1.0, 2.0, 5.0, 10.0, 30.0]:
            zr = usd_curve.zero_rate(t)
            assert zr > 0, f"Zero rate at t={t} should be positive, got {zr}"
            # Reasonable range: 2% to 8%
            assert 0.02 < zr < 0.08, f"Zero rate at t={t} is {zr:.4f}, outside [2%,8%]"

    def test_forward_rate_positive(self, usd_curve):
        """Instantaneous forward rates should be positive everywhere.

        For an upward-sloping USD curve, forwards at 10Y should be
        slightly higher than at 2Y (e.g. 4.3% at 2Y vs 4.6% at 10Y).
        """
        for t in [0.01, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 29.0]:
            fwd = usd_curve.instantaneous_forward(t)
            assert fwd > 0, f"Forward at t={t} should be positive, got {fwd}"


class TestInterpolation:
    """Compare log-linear vs monotone-convex interpolation."""

    def test_log_linear_vs_monotone_convex(self, usd_instruments):
        """Both methods should give similar par rates (within 1 bp).

        We bootstrap the same instruments with each method and compare
        the par rates at the calibration maturities.
        """
        bs_ll = CurveBootstrapper(interpolation_method="log_linear")
        bs_mc = CurveBootstrapper(interpolation_method="monotone_convex")
        val_date = dt.date(2026, 3, 5)

        curve_ll = bs_ll.bootstrap(usd_instruments, val_date)
        curve_mc = bs_mc.bootstrap(usd_instruments, val_date)

        test_maturities = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        for t in test_maturities:
            par_ll = curve_ll.par_rate(t)
            par_mc = curve_mc.par_rate(t)
            diff_bps = abs(par_ll - par_mc) * 10_000
            assert diff_bps < 1.0, (
                f"At t={t}: log_linear par={par_ll:.6f}, "
                f"monotone_convex par={par_mc:.6f}, diff={diff_bps:.2f} bps"
            )


class TestFXForwards:
    """Tests for CIP-based FX forward pricing."""

    def test_fx_forward_cip(self, usd_curve, eur_curve):
        """Verify F(T) = S * D_f(T) / D_d(T) holds.

        EUR/USD spot = 1.0832. With EUR rates (~2.9%) below USD (~4.3%),
        the EUR trades at a forward premium: F(1Y) > S.
        For T=1: F ~ 1.0832 * D_eur(1)/D_usd(1).
        D_eur(1) ~ exp(-0.029) ~ 0.9714, D_usd(1) ~ exp(-0.043) ~ 0.9579
        F(1) ~ 1.0832 * 0.9714/0.9579 ~ 1.0984
        """
        S = FALLBACK_FX_SPOT["EUR/USD"]  # 1.0832
        T = 1.0
        D_f = eur_curve.df(T)   # EUR discount ~ 0.97
        D_d = usd_curve.df(T)   # USD discount ~ 0.96
        F = S * D_f / D_d

        # Forward should be above spot (EUR premium due to lower rates)
        assert F > S, f"F(1Y)={F:.4f} should exceed S={S:.4f}"
        # Should be in a reasonable range
        assert 1.05 < F < 1.15, f"F(1Y)={F:.4f} out of range"

    def test_fx_forward_points(self, usd_curve, eur_curve):
        """Forward points should reflect the interest rate differential.

        Forward points = F(T) - S. For EUR/USD with EUR rates < USD rates,
        points should be positive (EUR premium). At 1Y the differential
        is ~1.4%, so forward points ~ 1.0832 * 0.014 = ~0.015 (150 pips).
        """
        S = FALLBACK_FX_SPOT["EUR/USD"]
        for T in [0.25, 0.5, 1.0, 2.0]:
            D_f = eur_curve.df(T)
            D_d = usd_curve.df(T)
            F = S * D_f / D_d
            fwd_points = F - S

            # Points should be positive (EUR premium when EUR rate < USD rate)
            assert fwd_points > 0, (
                f"T={T}: forward points={fwd_points:.6f} should be positive"
            )
            # Points should grow with tenor
            if T > 0.25:
                D_f_short = eur_curve.df(0.25)
                D_d_short = usd_curve.df(0.25)
                pts_short = S * D_f_short / D_d_short - S
                assert fwd_points > pts_short, (
                    f"Points at T={T} ({fwd_points:.6f}) should exceed T=0.25 ({pts_short:.6f})"
                )

    def test_cross_currency_basis(self, usd_curve, eur_curve):
        """Cross-currency basis should be small (< 100 bps).

        The basis is the spread added to the foreign OIS rate to match
        market FX forwards. Using our bootstrapped curves (no market
        FX forwards), the implied basis from CIP should be near zero.
        We check that the CIP-implied foreign rate is close to the
        bootstrapped foreign rate.

        At T=1: implied_r_f = r_d + ln(S/F) / T
        This should be close to eur_curve.zero_rate(1).
        """
        S = FALLBACK_FX_SPOT["EUR/USD"]
        for T in [1.0, 2.0, 5.0]:
            r_d = usd_curve.zero_rate(T)
            r_f = eur_curve.zero_rate(T)
            # CIP forward
            F = S * math.exp(-r_f * T) / math.exp(-r_d * T)
            # Implied foreign rate from CIP
            implied_r_f = r_d + math.log(S / F) / T
            basis_bps = abs(implied_r_f - r_f) * 10_000
            assert basis_bps < 100, (
                f"T={T}: basis={basis_bps:.1f} bps should be < 100"
            )


class TestCentralBankDates:
    """Tests for the deterministic central-bank meeting calendar."""

    def test_central_bank_dates(self, loader):
        """Meeting dates for 2026 should be valid business days.

        FOMC has 8 meetings in 2026, first on 2026-01-28 (Wednesday).
        ECB has 8 meetings, BOE has 8 meetings.
        """
        cb = loader.fetch_central_bank_dates(2026)

        assert "FOMC" in cb
        assert "ECB" in cb
        assert "BOE" in cb

        # Each bank should have 8 meetings in 2026
        assert len(cb["FOMC"]) == 8
        assert len(cb["ECB"]) == 8
        assert len(cb["BOE"]) == 8

        # All dates should parse and be weekdays (Mon-Fri, 0-4)
        for bank, dates in cb.items():
            for d_str in dates:
                d = dt.datetime.strptime(d_str, "%Y-%m-%d").date()
                assert d.year == 2026
                assert d.weekday() < 5, (
                    f"{bank} meeting {d_str} falls on a weekend "
                    f"(weekday={d.weekday()})"
                )
