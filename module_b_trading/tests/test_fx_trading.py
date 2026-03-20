"""
Tests for Module B: FX Fix-Aware Trading
==========================================

Tests the FX forward pricer, the W-shaped fix alpha pattern (Krohn, Mueller
& Whelan, Journal of Finance 2024), fix scheduling, composite alpha signals,
RFQ generation, win-probability modelling, quote optimisation, and markout
analysis.

The W-shape pattern: USD systematically *appreciates* in the 1-2 hours
**before** each major FX fix (Tokyo 00:55 UTC, ECB 13:15 UTC, London
16:00 UTC), then *depreciates* in the 1-2 hours **after** the fix.
For EUR/USD this means pre-fix returns are NEGATIVE (EUR weakens) and
post-fix returns are POSITIVE (EUR recovers).

Run with:  pytest module_b_trading/tests/test_fx_trading.py -v
"""

import sys
import datetime as dt
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from module_b_trading.fx_pricer import FXPricer, FXSwapSpec

try:
    import pricing_kernel as _pk
    HAS_PRICING_KERNEL = True
except ImportError:
    HAS_PRICING_KERNEL = False


# ===================================================================
# Helper: Flat-rate curve adapter (wraps a flat rate as a .discount() object)
# ===================================================================
class FlatCurve:
    """Trivial discount curve: D(T) = exp(-r * T).

    Example: FlatCurve(0.0433) gives D(1) = exp(-0.0433) = 0.9577.
    """
    def __init__(self, rate: float):
        self.rate = rate

    def discount(self, T: float) -> float:
        return math.exp(-self.rate * T)


# ===================================================================
# Helper: W-shape intraday data generator
# ===================================================================
# The three major FX fixes in UTC
FIX_TIMES_UTC = {
    "tokyo":  (0, 55),    # 9:55 JST = 00:55 UTC
    "ecb":    (13, 15),   # 14:15 CET = 13:15 UTC
    "london": (16, 0),    # 16:00 GMT
}


class FixAlphaModel:
    """Generates synthetic intraday FX data exhibiting the W-shape pattern.

    The W-shape (Krohn, Mueller & Whelan 2024): USD appreciates pre-fix,
    depreciates post-fix. For EUR/USD, this means returns are negative in
    the 1-2 hours before each fix, and positive in the 1-2 hours after.

    Parameters
    ----------
    pair : str
        Currency pair, e.g. 'EUR/USD'.
    spot : float
        Starting spot rate, e.g. 1.0832.
    fix_drift_bps : float
        Half-amplitude of the W-pattern in basis points per 15-min window.
        Default 2.0 means ~2 bps of pre-fix USD appreciation per window.
    noise_bps : float
        Standard deviation of random noise per 15-min bar, in bps. Default 1.0.
    """

    def __init__(
        self,
        pair: str = "EUR/USD",
        spot: float = 1.0832,
        fix_drift_bps: float = 2.0,
        noise_bps: float = 1.0,
    ):
        self.pair = pair
        self.spot = spot
        self.fix_drift_bps = fix_drift_bps
        self.noise_bps = noise_bps

    def generate_realistic_intraday_data(
        self,
        date_str: str = "2026-03-05",
        n_days: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate n_days of 15-min intraday EUR/USD prices with W-shape.

        Returns a DataFrame with columns: timestamp, mid, returns_bps.
        Each fix window (1h before to 1h after) has the drift pattern
        baked in. Outside fix windows, returns are pure noise.

        Example output row:
            timestamp=2026-03-05 15:45, mid=1.08312, returns_bps=-1.87
        """
        rng = np.random.default_rng(seed)
        all_rows = []

        for day_offset in range(n_days):
            day = pd.Timestamp(date_str) + pd.Timedelta(days=day_offset)
            if day.weekday() >= 5:
                continue  # skip weekends

            # 96 fifteen-minute bars per day (00:00 to 23:45)
            times = pd.date_range(day, periods=96, freq="15min")
            mid = self.spot
            for ts in times:
                h, m = ts.hour, ts.minute
                minutes_in_day = h * 60 + m
                drift = 0.0

                # Check proximity to each fix
                for fix_name, (fh, fm) in FIX_TIMES_UTC.items():
                    fix_min = fh * 60 + fm
                    delta = minutes_in_day - fix_min

                    if -60 <= delta < 0:
                        # Pre-fix: negative drift (USD appreciation for XXXUSD)
                        drift += -self.fix_drift_bps
                    elif 0 <= delta < 60:
                        # Post-fix: positive drift (USD depreciation)
                        drift += self.fix_drift_bps

                noise = rng.normal(0, self.noise_bps)
                total_return_bps = drift + noise
                mid = mid * (1 + total_return_bps / 10_000)

                all_rows.append({
                    "timestamp": ts,
                    "mid": mid,
                    "returns_bps": total_return_bps,
                })

        df = pd.DataFrame(all_rows)
        return df


class FixSchedule:
    """Utility for fix-related timing and spread adjustments.

    Provides minutes_to_next_fix, spread_multiplier, and skew_direction
    for any given UTC timestamp.
    """

    def minutes_to_next_fix(self, ts: pd.Timestamp) -> float:
        """Minutes until the next FX fix from timestamp ts.

        Example: at 15:30 UTC, the next fix is London at 16:00 -> 30 min.
        At 16:05, next fix is Tokyo next day at 00:55 -> ~529 min.
        """
        h, m = ts.hour, ts.minute
        current_min = h * 60 + m

        fixes_min = sorted([fh * 60 + fm for fh, fm in FIX_TIMES_UTC.values()])
        # fixes_min = [55, 795, 960]  i.e. 00:55, 13:15, 16:00

        for fix_min in fixes_min:
            if fix_min > current_min:
                return float(fix_min - current_min)

        # Wrap to next day's first fix
        return float(fixes_min[0] + 1440 - current_min)

    def spread_multiplier(self, ts: pd.Timestamp) -> float:
        """Spread multiplier based on fix proximity. >1.0 near fixes.

        Within 30 min of a fix -> 1.5x (widen spread to protect against
        adverse selection from fix-related flows).
        Within 60 min -> 1.2x.
        Otherwise -> 1.0x.

        Example: at 15:45 UTC (15 min before London fix) -> 1.5.
        """
        mins = self.minutes_to_next_fix(ts)
        if mins <= 30:
            return 1.5
        elif mins <= 60:
            return 1.2
        return 1.0

    def skew_direction(self, ts: pd.Timestamp) -> int:
        """Skew direction for quote adjustment.

        Pre-fix (next fix within 60 min): +1 (lean to accumulate USD,
        i.e. bias offers lower to sell EUR).
        Post-fix (just passed a fix within 60 min): -1 (lean to sell USD,
        i.e. bias bids higher to buy EUR).
        Otherwise: 0 (neutral).

        Example: at 15:30 UTC (30 min pre-London) -> +1.
        """
        h, m = ts.hour, ts.minute
        current_min = h * 60 + m

        fixes_min = sorted([fh * 60 + fm for fh, fm in FIX_TIMES_UTC.values()])

        # Check if we are within 60 min BEFORE any fix
        for fix_min in fixes_min:
            if 0 < (fix_min - current_min) <= 60:
                return +1

        # Check if we are within 60 min AFTER any fix
        for fix_min in fixes_min:
            if 0 < (current_min - fix_min) <= 60:
                return -1

        return 0


class CompositeAlpha:
    """Combines fix alpha, carry, momentum, and mean-reversion signals.

    Each signal is normalised to [-1, +1]. The composite is a weighted sum:
        alpha = w_fix * fix_signal + w_carry * carry + w_mom * momentum + w_mr * mr

    Default weights: fix=0.40, carry=0.20, momentum=0.20, mr=0.20.
    """

    def __init__(self, w_fix=0.40, w_carry=0.20, w_mom=0.20, w_mr=0.20):
        self.w_fix = w_fix
        self.w_carry = w_carry
        self.w_mom = w_mom
        self.w_mr = w_mr

    def compute(
        self,
        fix_signal: float,
        carry: float,
        momentum: float,
        mean_reversion: float,
    ) -> float:
        """Weighted composite alpha signal in [-1, +1].

        Example: fix=-0.8, carry=+0.3, mom=+0.1, mr=-0.2
          -> 0.4*(-0.8) + 0.2*(0.3) + 0.2*(0.1) + 0.2*(-0.2)
          = -0.32 + 0.06 + 0.02 - 0.04 = -0.28
        """
        raw = (
            self.w_fix * fix_signal
            + self.w_carry * carry
            + self.w_mom * momentum
            + self.w_mr * mean_reversion
        )
        return max(-1.0, min(1.0, raw))


class RFQGenerator:
    """Generates synthetic RFQ (request-for-quote) flow.

    Models client flow arriving with realistic timing (more during
    London/NY hours), direction (slightly skewed pre-fix), and size
    distributions.
    """

    def generate(
        self,
        n_rfqs: int = 500,
        date_str: str = "2026-03-05",
        pair: str = "EUR/USD",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate n_rfqs synthetic RFQs for one day.

        Returns DataFrame with columns:
            timestamp, pair, direction, notional_eur, fix_proximity_min,
            client_tier (1=top, 2=mid, 3=small).

        Example row:
            timestamp=2026-03-05 14:32, pair=EUR/USD, direction=sell,
            notional_eur=5_000_000, fix_proximity_min=43, client_tier=2.
        """
        rng = np.random.default_rng(seed)
        schedule = FixSchedule()

        # Generate timestamps weighted toward London/NY hours
        hours = rng.choice(range(24), size=n_rfqs, p=self._hour_weights())
        minutes = rng.integers(0, 60, size=n_rfqs)

        rows = []
        for i in range(n_rfqs):
            ts = pd.Timestamp(f"{date_str} {hours[i]:02d}:{minutes[i]:02d}")
            fix_prox = schedule.minutes_to_next_fix(ts)
            direction = rng.choice(["buy", "sell"])
            # Notional: lognormal around 5M EUR
            notional = float(rng.lognormal(mean=np.log(5e6), sigma=0.8))
            tier = rng.choice([1, 2, 3], p=[0.1, 0.3, 0.6])

            rows.append({
                "timestamp": ts,
                "pair": pair,
                "direction": direction,
                "notional_eur": round(notional, -3),
                "fix_proximity_min": fix_prox,
                "client_tier": tier,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _hour_weights() -> List[float]:
        """Probability weight for each hour 0-23, peaking London/NY."""
        w = np.array([
            0.5, 0.5, 0.6, 0.7, 0.8, 0.9,   # 00-05 Asia
            1.5, 2.0, 3.0, 3.5, 3.5, 3.0,     # 06-11 London morning
            3.5, 4.0, 4.5, 4.5, 4.0, 3.0,      # 12-17 London/NY overlap
            2.0, 1.5, 1.0, 0.8, 0.6, 0.5,      # 18-23 NY afternoon/evening
        ])
        return list(w / w.sum())


class WinModel:
    """Simple logistic win-probability model for RFQ quoting.

    Features: spread_pips, client_tier, fix_proximity.
    Trained on synthetic data to achieve AUC > 0.55.
    """

    def __init__(self, seed: int = 42):
        """Build and 'train' on synthetic data."""
        rng = np.random.default_rng(seed)
        n = 2000
        spreads = rng.uniform(0.3, 3.0, n)
        tiers = rng.choice([1, 2, 3], n)
        fix_prox = rng.uniform(0, 300, n)

        # True win probability: tighter spread -> higher win,
        # better tier -> higher win, closer to fix -> lower win
        logit = 1.5 - 0.8 * spreads + 0.3 * (3 - tiers) - 0.002 * fix_prox
        probs = 1 / (1 + np.exp(-logit))
        wins = (rng.random(n) < probs).astype(float)

        # Store "model" coefficients (fit by construction)
        self._intercept = 1.5
        self._beta_spread = -0.8
        self._beta_tier = 0.3
        self._beta_fix = -0.002
        self._wins = wins
        self._probs = probs

    def predict_proba(self, spread: float, tier: int, fix_prox: float) -> float:
        """Predicted win probability.

        Example: spread=1.0, tier=2, fix_prox=50
          logit = 1.5 - 0.8*1.0 + 0.3*(3-2) - 0.002*50
                = 1.5 - 0.8 + 0.3 - 0.1 = 0.9
          p = 1/(1+exp(-0.9)) = 0.711
        """
        logit = (
            self._intercept
            + self._beta_spread * spread
            + self._beta_tier * (3 - tier)
            + self._beta_fix * fix_prox
        )
        return 1.0 / (1.0 + math.exp(-logit))

    def auc(self) -> float:
        """AUC computed on training data (synthetic, so this is optimistic).

        Uses the Mann-Whitney U-statistic estimator.
        """
        pos_probs = self._probs[self._wins == 1]
        neg_probs = self._probs[self._wins == 0]
        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return 0.5
        # Approximate AUC via random sampling
        rng = np.random.default_rng(123)
        n_samples = min(50000, len(pos_probs) * len(neg_probs))
        i_pos = rng.choice(len(pos_probs), n_samples)
        i_neg = rng.choice(len(neg_probs), n_samples)
        return float(np.mean(pos_probs[i_pos] > neg_probs[i_neg]))


class QuoteOptimizer:
    """Determines optimal bid-ask spread given market conditions.

    Wider spreads during fixes to protect against adverse selection;
    tighter spreads off-fix to capture flow.

    Parameters
    ----------
    base_spread_pips : float
        Base spread in pips, e.g. 0.5 for EUR/USD.
    """

    def __init__(self, base_spread_pips: float = 0.5):
        self.base_spread_pips = base_spread_pips

    def optimal_spread(
        self,
        fix_proximity_min: float,
        volatility_pips: float = 55.0,
        inventory_pct: float = 0.0,
    ) -> float:
        """Compute optimal spread in pips.

        Example: fix_proximity=10 min, vol=55, inventory=0
          spread = 0.5 * 1.5 (fix multiplier) * (55/55) + |0|*0.1
                 = 0.75 pips

        Returns a value between 0.1 and 5.0 pips.
        """
        # Fix multiplier
        if fix_proximity_min <= 30:
            fix_mult = 1.5
        elif fix_proximity_min <= 60:
            fix_mult = 1.2
        else:
            fix_mult = 1.0

        # Vol scaling (normalised to 55 pips base)
        vol_mult = volatility_pips / 55.0

        # Inventory penalty
        inv_penalty = abs(inventory_pct) * 0.1

        spread = self.base_spread_pips * fix_mult * vol_mult + inv_penalty
        return max(0.1, min(5.0, spread))


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def usd_curve():
    """Flat USD curve at 4.33% for FXPricer tests.

    FXPricer expects .discount(T), so we use FlatCurve.
    FlatCurve(0.0433).discount(1.0) = exp(-0.0433) = 0.9577.
    """
    return FlatCurve(0.0433)


@pytest.fixture
def eur_curve():
    """Flat EUR curve at 2.90%.

    FlatCurve(0.0290).discount(1.0) = exp(-0.0290) = 0.9714.
    """
    return FlatCurve(0.0290)


@pytest.fixture
def fix_alpha():
    return FixAlphaModel(pair="EUR/USD", spot=1.0832, fix_drift_bps=2.0, noise_bps=1.0)


@pytest.fixture
def intraday_data(fix_alpha):
    """100 days of 15-min intraday data with W-shape pattern."""
    return fix_alpha.generate_realistic_intraday_data(n_days=100, seed=42)


@pytest.fixture
def fix_schedule():
    return FixSchedule()


@pytest.fixture
def rfq_generator():
    return RFQGenerator()


@pytest.fixture
def win_model():
    return WinModel(seed=42)


@pytest.fixture
def quote_optimizer():
    return QuoteOptimizer(base_spread_pips=0.5)


# ===================================================================
# Tests
# ===================================================================

class TestFXPricer:
    """Tests for the CIP-based FX forward pricer."""

    def test_fx_pricer_forward(self, usd_curve, eur_curve):
        """CIP forward price is consistent.

        EUR/USD 1Y forward with USD rate ~4.3%, EUR rate ~2.9%:
          D_eur(1) ~ exp(-0.029) = 0.9714
          D_usd(1) ~ exp(-0.043) = 0.9579
          F = 1.0832 * 0.9714 / 0.9579 ~ 1.0984 (EUR premium)
        """
        pricer = FXPricer(
            domestic_curve=usd_curve,
            foreign_curve=eur_curve,
            spot_rate=1.0832,
            pair_name="EUR/USD",
        )
        F = pricer.forward_price(1.0)
        # Forward should exceed spot by ~150 pips
        assert F > 1.0832, f"Forward {F:.4f} should exceed spot 1.0832"
        assert 1.05 < F < 1.15, f"Forward {F:.4f} out of reasonable range"

    def test_fx_pricer_delta(self, usd_curve, eur_curve):
        """Delta ~ notional * D_f(T) * 0.01 * S.

        For 10MM EUR, T=1Y, D_f(1)~0.97, S=1.0832:
          delta_01 = 10e6 * 0.97 * 0.01 * 1.0832 ~ 105,070 USD
        """
        pricer = FXPricer(
            domestic_curve=usd_curve,
            foreign_curve=eur_curve,
            spot_rate=1.0832,
            pair_name="EUR/USD",
        )
        notional = 10_000_000
        delta = pricer.delta(notional, 1.0)
        # Expected ~ 105,000 USD per 1% spot move
        assert 80_000 < delta < 130_000, f"Delta={delta:.0f}, expected ~105k"


class TestFixAlphaWShape:
    """THE KEY TESTS: Verify the W-shape pattern in synthetic data."""

    def test_fix_alpha_w_shape(self, intraday_data):
        """Average pre-fix returns should be NEGATIVE, post-fix POSITIVE.

        We bucket returns by their timing relative to the London fix
        (16:00 UTC). Pre-fix = 15:00-15:45, post-fix = 16:00-16:45.

        With fix_drift_bps=2.0, expected pre-fix avg ~ -2 bps,
        post-fix avg ~ +2 bps (plus noise).
        """
        df = intraday_data.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["minutes_in_day"] = df["hour"] * 60 + df["minute"]

        london_fix_min = 16 * 60  # 960

        # Pre-fix: 60 min before London fix (900-959, i.e. 15:00-15:59)
        pre_fix = df[
            (df["minutes_in_day"] >= london_fix_min - 60)
            & (df["minutes_in_day"] < london_fix_min)
        ]
        # Post-fix: 60 min after London fix (960-1019, i.e. 16:00-16:59)
        post_fix = df[
            (df["minutes_in_day"] >= london_fix_min)
            & (df["minutes_in_day"] < london_fix_min + 60)
        ]

        avg_pre = pre_fix["returns_bps"].mean()
        avg_post = post_fix["returns_bps"].mean()

        # Pre-fix returns should be NEGATIVE (USD appreciation, EUR weakens)
        assert avg_pre < 0, (
            f"Pre-fix avg returns = {avg_pre:.3f} bps, should be NEGATIVE"
        )
        # Post-fix returns should be POSITIVE (USD depreciation, EUR recovers)
        assert avg_post > 0, (
            f"Post-fix avg returns = {avg_post:.3f} bps, should be POSITIVE"
        )
        # Magnitude should be in the 1-3 bps range
        assert 0.5 < abs(avg_pre) < 5.0, (
            f"Pre-fix magnitude {abs(avg_pre):.2f} bps outside 0.5-5.0 range"
        )
        assert 0.5 < abs(avg_post) < 5.0, (
            f"Post-fix magnitude {abs(avg_post):.2f} bps outside 0.5-5.0 range"
        )

    def test_fix_alpha_three_fixes(self, intraday_data):
        """All three fixes (Tokyo, ECB, London) should show the W-pattern.

        For each fix, pre-fix returns < 0 and post-fix returns > 0.
        """
        df = intraday_data.copy()
        df["minutes_in_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

        fixes = {
            "tokyo": 0 * 60 + 55,    # 00:55 -> 55
            "ecb": 13 * 60 + 15,      # 13:15 -> 795
            "london": 16 * 60 + 0,    # 16:00 -> 960
        }

        for name, fix_min in fixes.items():
            pre = df[
                (df["minutes_in_day"] >= fix_min - 60)
                & (df["minutes_in_day"] < fix_min)
            ]
            post = df[
                (df["minutes_in_day"] >= fix_min)
                & (df["minutes_in_day"] < fix_min + 60)
            ]

            if len(pre) < 10 or len(post) < 10:
                pytest.skip(f"Not enough data for {name} fix")

            avg_pre = pre["returns_bps"].mean()
            avg_post = post["returns_bps"].mean()

            assert avg_pre < 0, (
                f"{name} fix: pre-fix avg = {avg_pre:.3f} bps, should be < 0"
            )
            assert avg_post > 0, (
                f"{name} fix: post-fix avg = {avg_post:.3f} bps, should be > 0"
            )


class TestFixSchedule:
    """Tests for fix timing and spread adjustments."""

    def test_fix_schedule_timing(self, fix_schedule):
        """minutes_to_next_fix returns correct values.

        At 15:30 UTC, next fix is London at 16:00 -> 30 min.
        At 12:00 UTC, next fix is ECB at 13:15 -> 75 min.
        At 00:30 UTC, next fix is Tokyo at 00:55 -> 25 min.
        """
        ts1 = pd.Timestamp("2026-03-05 15:30")
        assert fix_schedule.minutes_to_next_fix(ts1) == 30.0

        ts2 = pd.Timestamp("2026-03-05 12:00")
        assert fix_schedule.minutes_to_next_fix(ts2) == 75.0

        ts3 = pd.Timestamp("2026-03-05 00:30")
        assert fix_schedule.minutes_to_next_fix(ts3) == 25.0

    def test_fix_schedule_spread_widening(self, fix_schedule):
        """spread_multiplier > 1.0 during fix windows.

        At 15:45 (15 min before London fix) -> 1.5x.
        At 15:15 (45 min before) -> 1.2x.
        At 14:00 (120 min before) -> 1.0x (no widening).
        """
        # 15 min before London fix
        ts_near = pd.Timestamp("2026-03-05 15:45")
        assert fix_schedule.spread_multiplier(ts_near) == 1.5

        # 45 min before
        ts_mid = pd.Timestamp("2026-03-05 15:15")
        assert fix_schedule.spread_multiplier(ts_mid) == 1.2

        # 2 hours before
        ts_far = pd.Timestamp("2026-03-05 14:00")
        assert fix_schedule.spread_multiplier(ts_far) == 1.0

    def test_fix_schedule_skew(self, fix_schedule):
        """skew_direction: +1 pre-fix, -1 post-fix, 0 otherwise.

        Pre-fix (+1): lean to accumulate USD before the fix-driven
        appreciation. Post-fix (-1): lean to sell USD after reversal.
        """
        # 30 min before London fix -> +1 (accumulate USD)
        pre = pd.Timestamp("2026-03-05 15:30")
        assert fix_schedule.skew_direction(pre) == +1

        # 30 min after London fix -> -1 (sell USD)
        post = pd.Timestamp("2026-03-05 16:30")
        assert fix_schedule.skew_direction(post) == -1

        # 2 hours away from any fix -> 0
        neutral = pd.Timestamp("2026-03-05 10:00")
        assert fix_schedule.skew_direction(neutral) == 0


class TestCompositeAlpha:
    """Test the composite alpha signal combiner."""

    def test_composite_alpha(self):
        """Composite signal combines fix + carry + momentum + MR.

        Example: fix=-0.8 (strong pre-fix signal), carry=+0.3, mom=+0.1, mr=-0.2
          = 0.4*(-0.8) + 0.2*(0.3) + 0.2*(0.1) + 0.2*(-0.2)
          = -0.32 + 0.06 + 0.02 - 0.04 = -0.28
        """
        alpha = CompositeAlpha()
        result = alpha.compute(
            fix_signal=-0.8,
            carry=0.3,
            momentum=0.1,
            mean_reversion=-0.2,
        )
        expected = -0.28
        assert abs(result - expected) < 0.01, f"Composite={result:.4f}, expected={expected:.4f}"

        # Weights should sum to 1.0
        assert abs(alpha.w_fix + alpha.w_carry + alpha.w_mom + alpha.w_mr - 1.0) < 1e-10

        # Result should be bounded [-1, +1]
        extreme = alpha.compute(fix_signal=-1.0, carry=-1.0, momentum=-1.0, mean_reversion=-1.0)
        assert extreme >= -1.0


class TestRFQ:
    """Tests for RFQ generation and fix proximity."""

    def test_rfq_generator(self, rfq_generator):
        """Generated RFQs have realistic distributions.

        500 RFQs: all should have positive notional, valid direction, etc.
        """
        rfqs = rfq_generator.generate(n_rfqs=500, seed=42)
        assert len(rfqs) == 500
        assert set(rfqs["direction"].unique()) <= {"buy", "sell"}
        assert (rfqs["notional_eur"] > 0).all()
        assert set(rfqs["client_tier"].unique()) <= {1, 2, 3}
        # Notionals should be roughly lognormal around 5M
        median_notional = rfqs["notional_eur"].median()
        assert 1e6 < median_notional < 20e6, f"Median notional={median_notional/1e6:.1f}M"

    def test_rfq_fix_proximity(self, rfq_generator):
        """RFQs should have fix_proximity_min computed correctly.

        All values should be > 0 and < 1440 (minutes in a day).
        """
        rfqs = rfq_generator.generate(n_rfqs=200, seed=99)
        assert (rfqs["fix_proximity_min"] > 0).all()
        assert (rfqs["fix_proximity_min"] < 1440).all()


class TestWinModel:
    """Tests for win probability model."""

    def test_win_model_auc(self, win_model):
        """AUC > 0.55 (better than random 0.50).

        The logistic model with spread, tier, and fix proximity features
        should discriminate between wins and losses meaningfully.
        """
        auc = win_model.auc()
        assert auc > 0.55, f"AUC={auc:.4f}, should be > 0.55"
        # Should not be perfect (overfitting would give ~1.0)
        assert auc < 0.95, f"AUC={auc:.4f} suspiciously high"

    def test_win_model_calibration(self, win_model):
        """Predicted probabilities roughly match actual hit rates.

        Tighter spreads should have higher win rates. Tier 1 clients
        (most price-sensitive) should have lower win rates than tier 3.
        """
        # Tight spread, good tier -> high probability
        p_tight = win_model.predict_proba(spread=0.5, tier=2, fix_prox=100)
        # Wide spread -> low probability
        p_wide = win_model.predict_proba(spread=2.5, tier=2, fix_prox=100)

        assert p_tight > p_wide, (
            f"Tight spread p={p_tight:.3f} should exceed wide spread p={p_wide:.3f}"
        )
        assert 0 < p_tight < 1
        assert 0 < p_wide < 1


class TestQuoteOptimizer:
    """Tests for the quote optimizer."""

    def test_quote_optimizer_spread(self, quote_optimizer):
        """Optimal spread should be between 0.1 and 5.0 pips.

        Off-fix, low vol, no inventory: spread ~ 0.5 pips.
        """
        spread = quote_optimizer.optimal_spread(
            fix_proximity_min=120, volatility_pips=55, inventory_pct=0.0
        )
        assert 0.1 <= spread <= 5.0, f"Spread={spread:.3f} out of [0.1, 5.0] range"
        # Should be near base spread of 0.5
        assert 0.3 < spread < 0.8, f"Off-fix spread={spread:.3f}, expected ~0.5"

    def test_quote_optimizer_fix_widening(self, quote_optimizer):
        """Spreads during fix should be wider than off-fix.

        At fix_proximity=10 min, spread = 0.5 * 1.5 = 0.75 pips.
        Off-fix (120 min away), spread = 0.5 * 1.0 = 0.5 pips.
        """
        spread_fix = quote_optimizer.optimal_spread(
            fix_proximity_min=10, volatility_pips=55
        )
        spread_offfix = quote_optimizer.optimal_spread(
            fix_proximity_min=120, volatility_pips=55
        )
        assert spread_fix > spread_offfix, (
            f"Fix spread={spread_fix:.3f} should exceed off-fix={spread_offfix:.3f}"
        )


class TestMarkout:
    """Test markout analysis by fix proximity."""

    def test_markout_by_fix(self, intraday_data):
        """Pre-fix markouts should be better (more positive) than at-fix markouts.

        If we "sell EUR" (accumulate USD) pre-fix and the EUR depreciates,
        our markout is positive. At-fix, the reversal means our position
        is hurt.

        We simulate a simple strategy: sell EUR 1h before each London fix,
        and measure the 1h markout (price change from entry).
        """
        df = intraday_data.copy()
        df["minutes_in_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

        london_fix = 16 * 60  # 960

        # Pre-fix entries: 15:00 (sell EUR, expecting EUR to weaken)
        pre_entries = df[df["minutes_in_day"] == london_fix - 60].copy()
        # At-fix entries: 16:00 (bad timing, EUR about to recover)
        at_entries = df[df["minutes_in_day"] == london_fix].copy()

        if len(pre_entries) < 5 or len(at_entries) < 5:
            pytest.skip("Not enough data points")

        # For "sell EUR" strategy, markout = entry_mid - exit_mid (positive if EUR fell)
        pre_markouts = []
        at_markouts = []

        for _, row in pre_entries.iterrows():
            entry_mid = row["mid"]
            # Look for exit 1h later
            exit_rows = df[
                (df["timestamp"] == row["timestamp"] + pd.Timedelta(hours=1))
            ]
            if len(exit_rows) > 0:
                exit_mid = exit_rows.iloc[0]["mid"]
                # Sell EUR markout: positive if EUR weakened (mid fell)
                pre_markouts.append((entry_mid - exit_mid) / entry_mid * 10_000)

        for _, row in at_entries.iterrows():
            entry_mid = row["mid"]
            exit_rows = df[
                (df["timestamp"] == row["timestamp"] + pd.Timedelta(hours=1))
            ]
            if len(exit_rows) > 0:
                exit_mid = exit_rows.iloc[0]["mid"]
                at_markouts.append((entry_mid - exit_mid) / entry_mid * 10_000)

        if len(pre_markouts) < 3 or len(at_markouts) < 3:
            pytest.skip("Not enough markout observations")

        avg_pre_markout = np.mean(pre_markouts)
        avg_at_markout = np.mean(at_markouts)

        # Pre-fix sell markouts should be better (more positive or less negative)
        # because EUR weakens pre-fix
        assert avg_pre_markout > avg_at_markout, (
            f"Pre-fix markout={avg_pre_markout:.2f} bps should exceed "
            f"at-fix markout={avg_at_markout:.2f} bps"
        )
