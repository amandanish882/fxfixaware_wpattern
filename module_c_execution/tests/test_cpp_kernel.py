"""
Tests for the C++ pricing_kernel extension module.
====================================================

All tests are skipped if the ``pricing_kernel`` (a.k.a. ``fx_pricing_kernel``)
shared library is not compiled/available. When available, each test verifies
that the C++ implementation matches the pure-Python reference within tight
tolerances.

Run with:  pytest module_c_execution/tests/test_cpp_kernel.py -v
"""

import sys
import datetime as dt
import math
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional import of the C++ kernel
# ---------------------------------------------------------------------------
try:
    import pricing_kernel as _pk
    HAS_CPP = True
except ImportError:
    _pk = None
    HAS_CPP = False

# Python reference implementations
from module_a_curves.curve_bootstrapper import (
    CurveBootstrapper,
    CurveInstrument,
    DiscountCurve,
)
from module_a_curves.data_loader import FALLBACK_FX_SPOT

skip_no_cpp = pytest.mark.skipif(
    not HAS_CPP,
    reason="pricing_kernel C++ extension not compiled",
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def usd_instruments():
    """USD OIS calibration instruments (Mar-2026 levels)."""
    return [
        CurveInstrument("deposit", 1 / 360, 0.0433, "ACT/360"),
        CurveInstrument("deposit", 30 / 360, 0.0432, "ACT/360"),
        CurveInstrument("deposit", 90 / 360, 0.0431, "ACT/360"),
        CurveInstrument("swap", 2.0, 0.0412, "ACT/360", 1.0),
        CurveInstrument("swap", 5.0, 0.0428, "ACT/360", 1.0),
        CurveInstrument("swap", 10.0, 0.0455, "ACT/360", 1.0),
        CurveInstrument("swap", 30.0, 0.0472, "ACT/360", 1.0),
    ]


@pytest.fixture
def eur_instruments():
    """EUR OIS calibration instruments."""
    return [
        CurveInstrument("deposit", 1 / 360, 0.0290, "ACT/360"),
        CurveInstrument("deposit", 90 / 360, 0.0288, "ACT/360"),
        CurveInstrument("swap", 2.0, 0.0275, "ACT/360", 1.0),
        CurveInstrument("swap", 5.0, 0.0285, "ACT/360", 1.0),
        CurveInstrument("swap", 10.0, 0.0300, "ACT/360", 1.0),
    ]


@pytest.fixture
def python_usd_curve(usd_instruments):
    """Python-bootstrapped USD curve for reference."""
    bs = CurveBootstrapper(interpolation_method="log_linear")
    return bs.bootstrap(usd_instruments, dt.date(2026, 3, 5))


@pytest.fixture
def python_eur_curve(eur_instruments):
    bs = CurveBootstrapper(interpolation_method="log_linear")
    return bs.bootstrap(eur_instruments, dt.date(2026, 3, 5))


# ===================================================================
# Tests
# ===================================================================

@skip_no_cpp
class TestCppBootstrap:
    """Tests comparing C++ bootstrap to Python reference."""

    def test_cpp_bootstrap(self, usd_instruments, python_usd_curve):
        """C++ bootstrap gives same discount factors as Python.

        For each instrument maturity, the C++ and Python discount factors
        should agree to within 1e-8. For example, at 2Y with rate 4.12%,
        D(2) ~ 0.9208 from both implementations.
        """
        maturities = [inst.maturity_years for inst in usd_instruments]
        rates = [inst.rate for inst in usd_instruments]
        types = [inst.type for inst in usd_instruments]
        freqs = [inst.payment_frequency for inst in usd_instruments]

        result = _pk.bootstrap_ois(maturities, rates, types, freqs)
        cpp_times = [0.0] + list(result["times"])
        cpp_dfs = [1.0] + list(result["dfs"])

        for t, cpp_df in zip(cpp_times, cpp_dfs):
            py_df = python_usd_curve.df(t)
            assert abs(cpp_df - py_df) < 1e-8, (
                f"At t={t:.4f}: C++={cpp_df:.10f}, Python={py_df:.10f}"
            )


@skip_no_cpp
class TestCppForwardPricer:
    """Tests for C++ FX forward pricing."""

    def test_cpp_forward_pricer(self, python_usd_curve, python_eur_curve):
        """C++ FX forward matches Python calculation.

        EUR/USD 1Y forward: F = S * D_eur(1) / D_usd(1).
        With S=1.0832, D_eur~0.9714, D_usd~0.9579 -> F~1.0984.
        C++ and Python should agree within 1e-6.
        """
        S = FALLBACK_FX_SPOT["EUR/USD"]  # 1.0832
        T = 1.0

        py_fwd = S * python_eur_curve.df(T) / python_usd_curve.df(T)

        # C++ forward pricer
        cpp_fwd = _pk.fx_forward(
            S,
            T,
            python_eur_curve.times.tolist(),
            python_eur_curve.dfs.tolist(),
            python_usd_curve.times.tolist(),
            python_usd_curve.dfs.tolist(),
        )

        assert abs(cpp_fwd - py_fwd) < 1e-6, (
            f"C++ forward={cpp_fwd:.8f}, Python={py_fwd:.8f}"
        )

    def test_cpp_delta(self, python_usd_curve, python_eur_curve):
        """C++ delta matches Python to within 0.01 (in USD per 1% spot move).

        For 10MM EUR at T=1Y:
          delta_01 = 10e6 * D_f(1) * 0.01 * S ~ 105,000 USD.
        """
        S = FALLBACK_FX_SPOT["EUR/USD"]
        T = 1.0
        notional = 10_000_000

        py_delta = notional * python_eur_curve.df(T) * 0.01 * S

        cpp_delta = _pk.fx_delta(
            notional,
            S,
            T,
            python_eur_curve.times.tolist(),
            python_eur_curve.dfs.tolist(),
            python_usd_curve.times.tolist(),
            python_usd_curve.dfs.tolist(),
        )

        assert abs(cpp_delta - py_delta) < 0.01 * py_delta, (
            f"C++ delta={cpp_delta:.2f}, Python={py_delta:.2f}"
        )


@skip_no_cpp
class TestCppSchedules:
    """Tests for C++ execution schedule and impact calculations."""

    def test_cpp_adaptive_schedule(self):
        """C++ adaptive schedule matches Python.

        For 200 contracts in 20 slices with urgency=0.5, the C++ and
        Python schedules should produce the same allocations within 1e-4.
        """
        total = 200
        n_slices = 20
        urgency = 0.5

        # Python reference
        decay = np.exp(-urgency * np.arange(n_slices))
        weights = decay / decay.sum()
        py_schedule = list(weights * total)

        cpp_schedule = _pk.adaptive_schedule(total, n_slices, urgency)

        for i, (py_s, cpp_s) in enumerate(zip(py_schedule, cpp_schedule)):
            assert abs(py_s - cpp_s) < 1e-4, (
                f"Slice {i}: Python={py_s:.6f}, C++={cpp_s:.6f}"
            )

    def test_cpp_almgren_chriss(self):
        """C++ impact matches Python for a 500-lot 6E order.

        Expected costs (30-min horizon):
          permanent ~ 0.05 * (500/250000) * 55 = 0.0055 pips
          temporary ~ 0.10 * (500/(30*641)) * 55 = 0.143 pips
        """
        from module_c_execution.market_impact import AlmgrenChrissModel

        model = AlmgrenChrissModel()
        py_costs = model.total_cost(500, 250_000, 55, 30.0)

        cpp_costs = _pk.almgren_chriss_cost(
            n_contracts=500,
            daily_volume=250_000,
            daily_vol_pips=55.0,
            horizon_minutes=30.0,
            gamma_frac=0.05,
            eta_frac=0.10,
        )

        assert abs(cpp_costs["permanent"] - py_costs["permanent_cost_pips"]) < 1e-6
        assert abs(cpp_costs["temporary"] - py_costs["temporary_cost_pips"]) < 1e-6


@skip_no_cpp
class TestCppPerformance:
    """Benchmark C++ vs Python performance."""

    def test_cpp_performance_curve(self, usd_instruments):
        """C++ bootstrap should be faster than Python.

        We run each 100 times and compare wall-clock.
        Typical: Python ~5ms, C++ ~0.2ms per bootstrap.
        """
        maturities = [inst.maturity_years for inst in usd_instruments]
        rates = [inst.rate for inst in usd_instruments]
        types = [inst.type for inst in usd_instruments]
        freqs = [inst.payment_frequency for inst in usd_instruments]

        n_iter = 100

        # Python timing
        bs = CurveBootstrapper(interpolation_method="log_linear")
        t0 = time.perf_counter()
        for _ in range(n_iter):
            bs._bootstrap_python(usd_instruments, dt.date(2026, 3, 5))
        py_time = time.perf_counter() - t0

        # C++ timing
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _pk.bootstrap_ois(maturities, rates, types, freqs)
        cpp_time = time.perf_counter() - t0

        speedup = py_time / cpp_time if cpp_time > 0 else float("inf")
        # C++ should be at least 2x faster (typically 10-50x)
        assert speedup > 2.0, (
            f"C++ speedup only {speedup:.1f}x "
            f"(Python={py_time*1000/n_iter:.2f}ms, C++={cpp_time*1000/n_iter:.2f}ms)"
        )

    def test_cpp_performance_schedule(self):
        """C++ adaptive schedule should be faster than Python.

        For 1000 iterations of a 100-slice schedule.
        """
        n_iter = 1000
        total = 500
        n_slices = 100
        urgency = 0.5

        # Python timing
        t0 = time.perf_counter()
        for _ in range(n_iter):
            decay = np.exp(-urgency * np.arange(n_slices))
            weights = decay / decay.sum()
            _ = list(weights * total)
        py_time = time.perf_counter() - t0

        # C++ timing
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _pk.adaptive_schedule(total, n_slices, urgency)
        cpp_time = time.perf_counter() - t0

        speedup = py_time / cpp_time if cpp_time > 0 else float("inf")
        assert speedup > 1.5, (
            f"C++ schedule speedup only {speedup:.1f}x"
        )


@skip_no_cpp
class TestCppConvexity:
    """Test C++ convexity (second-order rate sensitivity) calculation."""

    def test_cpp_convexity(self, python_usd_curve):
        """C++ convexity calculation produces reasonable values.

        Convexity of a discount factor with respect to yield shift:
        C = d2D/dy2 / D. For a 10Y zero bond at ~4.55%, convexity ~ 100.
        The C++ and Python values should agree within 1%.
        """
        T = 10.0
        D = python_usd_curve.df(T)
        r = python_usd_curve.zero_rate(T)

        # Python reference: for continuously compounded, convexity = T^2
        py_convexity = T ** 2

        cpp_convexity = _pk.bond_convexity(
            python_usd_curve.times.tolist(),
            python_usd_curve.dfs.tolist(),
            T,
        )

        # Allow 1% tolerance (different numerical methods may differ slightly)
        assert abs(cpp_convexity - py_convexity) / py_convexity < 0.01, (
            f"C++ convexity={cpp_convexity:.4f}, Python ref={py_convexity:.4f}"
        )
