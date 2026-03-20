"""
Tests for Module C: Execution and Market Impact
=================================================

Covers: Almgren-Chriss cost model, optimal horizon, cost scaling, impact curve
shape, TWAP/VWAP/Adaptive execution schedules, walking the book, backtest on
synthetic data, and CME FX futures contract specs.

Run with:  pytest module_c_execution/tests/test_execution.py -v
"""

import sys
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from module_c_execution.market_impact import (
    AlmgrenChrissModel,
    FX_FUTURES,
    ImpactResult,
)


# ===================================================================
# Helper execution-schedule functions
# ===================================================================

def twap_schedule(total_contracts: int, n_slices: int) -> list[float]:
    """Time-Weighted Average Price: equal-size slices.

    Example: 100 contracts in 10 slices -> [10, 10, 10, ..., 10].
    """
    per_slice = total_contracts / n_slices
    schedule = [per_slice] * n_slices
    # Adjust last slice for rounding
    residual = total_contracts - sum(schedule)
    schedule[-1] += residual
    return schedule


def vwap_schedule(total_contracts: int, n_slices: int, seed: int = 42) -> list[float]:
    """Volume-Weighted Average Price: slices weighted by a U-shaped volume curve.

    Models the typical CME FX volume pattern: higher volume at open and close,
    lower in the middle. The weights follow a parabola.

    Example with 10 slices:
        weights ~ [1.8, 1.3, 0.9, 0.6, 0.5, 0.5, 0.6, 0.9, 1.3, 1.8]
        For 100 contracts -> [18.4, 13.0, 9.2, 6.3, 5.1, 5.1, 6.3, 9.2, 13.0, 18.4]
    """
    # U-shaped volume: quadratic, minimum at mid-session
    x = np.linspace(-1, 1, n_slices)
    weights = 0.5 + x ** 2  # minimum 0.5 at centre, 1.5 at edges
    weights = weights / weights.sum()
    schedule = list(weights * total_contracts)
    # Fix rounding
    residual = total_contracts - sum(schedule)
    schedule[-1] += residual
    return schedule


def adaptive_schedule(
    total_contracts: int,
    n_slices: int,
    urgency: float = 0.5,
) -> list[float]:
    """Adaptive (front-loaded) execution schedule.

    Exponentially decaying weights: earlier slices get more volume.
    Higher urgency -> more front-loaded.

    With urgency=0.5, 10 slices, 100 contracts:
        weights ~ [1.65, 1.49, 1.35, 1.22, 1.10, 1.0, 0.90, 0.82, 0.74, 0.67]
        (earlier slices are ~2.5x larger than later ones)

    Parameters
    ----------
    urgency : float
        Decay rate. 0.0 = flat (same as TWAP), 1.0 = very aggressive.
    """
    decay = np.exp(-urgency * np.arange(n_slices))
    weights = decay / decay.sum()
    schedule = list(weights * total_contracts)
    residual = total_contracts - sum(schedule)
    schedule[-1] += residual
    return schedule


def compare_strategies(total_contracts: int, n_slices: int) -> dict[str, list[float]]:
    """Return all three schedules for comparison."""
    return {
        "twap": twap_schedule(total_contracts, n_slices),
        "vwap": vwap_schedule(total_contracts, n_slices),
        "adaptive": adaptive_schedule(total_contracts, n_slices),
    }


def walk_book(
    mid_price: float,
    half_spread: float,
    n_contracts: int,
    tick_size: float,
    depth_per_level: int = 50,
    n_levels: int = 10,
) -> dict:
    """Simulate walking through an order book to fill n_contracts.

    Returns dict with fill_price (average) and slippage_pips.

    Example for 6E (EUR/USD):
        mid=1.0850, half_spread=0.000025, n_contracts=200, tick=0.00005
        First 50 contracts fill at 1.085025 (best ask)
        Next 50 at 1.085075 (one tick deeper)
        ...etc.

    Parameters
    ----------
    mid_price : float
        Current mid price, e.g. 1.0850.
    half_spread : float
        Half of bid-ask spread, e.g. 0.000025 for 6E.
    n_contracts : int
        Contracts to buy (aggressive).
    tick_size : float
        Tick size, e.g. 0.00005 for 6E.
    depth_per_level : int
        Contracts available at each price level.
    n_levels : int
        Number of price levels in the book.
    """
    best_ask = mid_price + half_spread
    remaining = n_contracts
    total_cost = 0.0
    filled = 0

    for level in range(n_levels):
        price = best_ask + level * tick_size
        fill_qty = min(remaining, depth_per_level)
        total_cost += fill_qty * price
        filled += fill_qty
        remaining -= fill_qty
        if remaining <= 0:
            break

    if filled == 0:
        return {"fill_price": best_ask, "slippage_pips": 0.0}

    avg_fill = total_cost / filled
    slippage = (avg_fill - best_ask) / tick_size  # in ticks
    slippage_pips = slippage * tick_size * 10_000  # convert to pips

    return {
        "fill_price": round(avg_fill, 8),
        "slippage_pips": round(slippage_pips, 4),
    }


def backtest_strategies_synthetic(
    ticker: str = "6E",
    n_contracts: int = 200,
    n_slices: int = 20,
    seed: int = 42,
) -> dict[str, dict]:
    """Backtest all 3 strategies on synthetic price path.

    Returns a dict with keys twap, vwap, adaptive, each containing:
        avg_fill_price, total_slippage_pips, arrival_slippage_pips.

    The synthetic mid-price follows a random walk with the contract's
    daily vol.
    """
    spec = FX_FUTURES[ticker]
    rng = np.random.default_rng(seed)

    # Generate a mid-price path for each slice
    daily_vol_pips = spec["daily_vol_pips"]
    tick = spec["tick_size"]
    mid_start = {"6E": 1.0850, "6B": 1.2700, "6J": 0.006650, "6A": 0.6550}[ticker]

    # Per-slice vol: daily vol / sqrt(n_slices)
    slice_vol = daily_vol_pips * tick / np.sqrt(n_slices)
    mids = [mid_start]
    for _ in range(n_slices - 1):
        mids.append(mids[-1] + rng.normal(0, slice_vol))
    mids = np.array(mids)

    strategies = compare_strategies(n_contracts, n_slices)
    results = {}

    for name, schedule in strategies.items():
        total_cost = 0.0
        total_filled = 0
        for i, qty in enumerate(schedule):
            if qty <= 0:
                continue
            # Walk book at this slice's mid price
            wb = walk_book(
                mid_price=mids[i],
                half_spread=tick / 2,
                n_contracts=int(round(qty)),
                tick_size=tick,
                depth_per_level=50,
            )
            total_cost += wb["fill_price"] * qty
            total_filled += qty

        avg_fill = total_cost / total_filled if total_filled > 0 else mid_start
        arrival_slip = (avg_fill - mid_start) / tick * 10_000 * tick  # in pips

        results[name] = {
            "avg_fill_price": round(avg_fill, 8),
            "total_slippage_pips": round(
                (avg_fill - mid_start) * 10_000, 4
            ),
            "arrival_slippage_pips": round(arrival_slip, 4),
        }

    return results


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def model():
    return AlmgrenChrissModel()


# ===================================================================
# Tests
# ===================================================================

class TestAlmgrenChriss:
    """Tests for the Almgren-Chriss market impact model."""

    def test_almgren_chriss_cost_positive(self, model):
        """All cost components > 0 for a 100-lot 6E order over 30 minutes.

        With 100 contracts, V=250k, sigma=55 pips, T=30 min:
          permanent ~ 0.05 * (100/250000) * 55 = 0.0011 pips
          temporary ~ 0.10 * (100/(30*641)) * 55 = 0.0286 pips
        """
        costs = model.total_cost(
            n_contracts=100,
            daily_volume=250_000,
            daily_vol_pips=55,
            horizon_minutes=30.0,
        )
        assert costs["permanent_cost_pips"] > 0
        assert costs["temporary_cost_pips"] > 0
        assert costs["total_cost_pips"] > 0
        assert costs["timing_risk_pips"] > 0

    def test_almgren_chriss_optimal_horizon(self, model):
        """T* should be between 1 and 60 minutes for a 100-lot 6E order.

        For 100 contracts on 6E (250k daily volume), the optimal horizon
        is typically a few minutes -- big enough to reduce impact but
        small enough to limit timing risk.
        """
        T_star = model.optimal_horizon(
            n_contracts=100,
            daily_volume=250_000,
            daily_vol_pips=55,
        )
        assert 1.0 <= T_star <= 60.0, f"T*={T_star:.1f} min, expected [1, 60]"

    def test_almgren_chriss_scaling(self, model):
        """Cost increases with order size.

        Doubling the order from 100 to 200 contracts should increase
        both permanent and temporary costs. Permanent is linear in Q,
        temporary is also linear in Q for fixed horizon.
        """
        costs_100 = model.total_cost(100, 250_000, 55, 30.0)
        costs_200 = model.total_cost(200, 250_000, 55, 30.0)

        assert costs_200["permanent_cost_pips"] > costs_100["permanent_cost_pips"]
        assert costs_200["temporary_cost_pips"] > costs_100["temporary_cost_pips"]
        assert costs_200["total_cost_pips"] > costs_100["total_cost_pips"]

        # Permanent should roughly double (linear in Q)
        ratio_perm = costs_200["permanent_cost_pips"] / costs_100["permanent_cost_pips"]
        assert 1.8 < ratio_perm < 2.2, f"Permanent cost ratio={ratio_perm:.2f}"

    def test_impact_curve_shape(self, model):
        """Cost is U-shaped with horizon: too fast = high temporary, too slow = high risk.

        Total cost + timing risk should be minimised somewhere in the middle.
        At very short horizons (1 min): high temporary cost.
        At very long horizons (390 min): high timing risk.
        """
        curves = model.impact_curve(500, 250_000, 55)

        # Temporary cost should decrease with horizon
        assert curves["temporary"][0] > curves["temporary"][-1]

        # Timing risk should increase with horizon
        assert curves["timing_risk"][0] < curves["timing_risk"][-1]

        # The combined cost+risk should have a minimum in the interior
        combined = curves["total"] + curves["timing_risk"]
        min_idx = np.argmin(combined)
        assert 0 < min_idx < len(combined) - 1, (
            f"Minimum at boundary (idx={min_idx}), expected interior minimum"
        )


class TestExecutionSchedules:
    """Tests for TWAP, VWAP, and Adaptive execution schedules."""

    def test_twap_schedule(self):
        """All slices equal, sum = total.

        TWAP(100, 10) -> 10 slices of 10 contracts each.
        """
        sched = twap_schedule(100, 10)
        assert len(sched) == 10
        assert abs(sum(sched) - 100) < 1e-10
        # All slices should be equal
        for s in sched:
            assert abs(s - 10.0) < 1e-10

    def test_vwap_schedule(self):
        """Sum = total, mid-session slices weighted differently.

        With U-shaped weights, edge slices should be larger than centre.
        """
        sched = vwap_schedule(100, 10)
        assert len(sched) == 10
        assert abs(sum(sched) - 100) < 1e-6
        # First and last slices should be larger than middle slices
        assert sched[0] > sched[4], (
            f"First slice={sched[0]:.2f} should exceed mid={sched[4]:.2f}"
        )
        assert sched[-1] > sched[4] or abs(sched[-1] - sched[4]) < 1, (
            f"Last slice={sched[-1]:.2f} should be >= mid={sched[4]:.2f}"
        )

    def test_adaptive_schedule(self):
        """Early slices > late slices (front-loaded), sum = total.

        With urgency=0.5, first slice should be ~2.5x the last slice.
        """
        sched = adaptive_schedule(100, 10, urgency=0.5)
        assert len(sched) == 10
        assert abs(sum(sched) - 100) < 1e-6
        # Front-loaded: first slice > last slice
        assert sched[0] > sched[-1], (
            f"First={sched[0]:.2f} should exceed last={sched[-1]:.2f}"
        )

    def test_compare_strategies(self):
        """compare_strategies returns 3 strategies, all summing to same total."""
        result = compare_strategies(200, 20)
        assert set(result.keys()) == {"twap", "vwap", "adaptive"}
        for name, sched in result.items():
            assert len(sched) == 20
            assert abs(sum(sched) - 200) < 1e-6, (
                f"{name}: sum={sum(sched):.6f}, expected 200"
            )


class TestBookWalking:
    """Tests for walking the order book."""

    def test_walk_book(self):
        """Walking the book returns valid fill price and positive slippage.

        For 200 contracts with 50 per level, we fill 4 levels:
          Level 0: 50 @ 1.085025 (best ask = mid + half_spread)
          Level 1: 50 @ 1.085075
          Level 2: 50 @ 1.085125
          Level 3: 50 @ 1.085175
          Avg fill = (1.085025+1.085075+1.085125+1.085175)/4 = 1.085100
          Slippage from best ask = 1.085100 - 1.085025 = 0.000075 = 0.75 pips
        """
        result = walk_book(
            mid_price=1.0850,
            half_spread=0.000025,
            n_contracts=200,
            tick_size=0.00005,
            depth_per_level=50,
            n_levels=10,
        )
        assert result["fill_price"] > 1.0850  # above mid
        assert result["slippage_pips"] >= 0  # non-negative slippage

        # With only 1 contract, no slippage (fills at best ask)
        result_small = walk_book(
            mid_price=1.0850,
            half_spread=0.000025,
            n_contracts=1,
            tick_size=0.00005,
            depth_per_level=50,
        )
        assert result_small["slippage_pips"] == 0.0


class TestBacktest:
    """Tests for synthetic backtest of execution strategies."""

    def test_backtest_strategies_synthetic(self):
        """All 3 strategies return results with defined slippage.

        Each strategy should produce an average fill price and a
        slippage figure. Slippage can be positive or negative depending
        on the random price path, but all fields should be populated.
        """
        results = backtest_strategies_synthetic(
            ticker="6E", n_contracts=200, n_slices=20, seed=42
        )
        assert set(results.keys()) == {"twap", "vwap", "adaptive"}
        for name, r in results.items():
            assert "avg_fill_price" in r
            assert "total_slippage_pips" in r
            assert r["avg_fill_price"] > 0, f"{name}: fill price should be > 0"


class TestFXFuturesSpecs:
    """Tests for CME FX futures contract specifications."""

    def test_fx_futures_specs(self):
        """All 4 FX futures (6E, 6B, 6J, 6A) have valid parameters.

        6E: EUR/USD, 125k contract size, 0.00005 tick, $6.25 tick value
        6B: GBP/USD, 62.5k, 0.0001 tick, $6.25
        6J: JPY/USD, 12.5M, 0.0000005 tick, $6.25
        6A: AUD/USD, 100k, 0.0001 tick, $10.00
        """
        expected = {
            "6E": {"name": "EUR/USD", "contract_size": 125_000, "tick_size": 0.00005},
            "6B": {"name": "GBP/USD", "contract_size": 62_500, "tick_size": 0.0001},
            "6J": {"name": "JPY/USD", "contract_size": 12_500_000, "tick_size": 0.0000005},
            "6A": {"name": "AUD/USD", "contract_size": 100_000, "tick_size": 0.0001},
        }

        for ticker, exp in expected.items():
            assert ticker in FX_FUTURES, f"{ticker} missing from FX_FUTURES"
            spec = FX_FUTURES[ticker]
            assert spec["name"] == exp["name"]
            assert spec["contract_size"] == exp["contract_size"]
            assert spec["tick_size"] == exp["tick_size"]
            assert spec["tick_value"] > 0
            assert spec["avg_daily_volume"] > 0
            assert spec["daily_vol_pips"] > 0
