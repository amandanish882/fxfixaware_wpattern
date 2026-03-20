"""
OIS Discount Curve Bootstrapper for USD and foreign-currency curves.

Sequential bootstrap algorithm:
- **Deposits**: D(t) = 1 / (1 + r * alpha)
  e.g. SOFR overnight at 4.33%, alpha=1/360:  D = 1/(1 + 0.0433/360) = 0.999880
- **Swaps**: D(t_n) = (1 - S_n * sum(alpha_j * D(t_j))) / (1 + S_n * alpha_n)
  Solved sequentially from short to long maturities.

Attempts to use the ``pricing_kernel`` C++ extension for performance;
falls back to pure-Python if unavailable.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from module_a_curves.interpolation import LogLinearInterpolator, MonotoneConvexInterpolator

logger = logging.getLogger(__name__)

# Try C++ pricing kernel for hot-path bootstrap
try:
    import pricing_kernel as _pk  # type: ignore[import-untyped]

    _USE_CPP = True
    logger.info("Using C++ pricing_kernel for bootstrap")
except ImportError:
    _pk = None
    _USE_CPP = False
    logger.debug("pricing_kernel not available; using pure-Python bootstrap")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CurveInstrument:
    """A single calibration instrument for the OIS curve.

    Parameters
    ----------
    type : str
        ``"deposit"`` or ``"swap"``.
    maturity_years : float
        Time to maturity in year fractions, e.g. 0.00278 (O/N), 2.0, 10.0.
    rate : float
        Quoted rate in decimal, e.g. 0.0433 for 4.33%.
    day_count : str
        Day-count convention, e.g. ``"ACT/360"`` or ``"ACT/365"``.
    payment_frequency : float
        Payment frequency in years: 1.0 for annual, 0.5 for semi-annual,
        0.25 for quarterly.  Ignored for deposits.
    """

    type: str  # "deposit" or "swap"
    maturity_years: float
    rate: float
    day_count: str = "ACT/360"
    payment_frequency: float = 1.0

    def alpha(self, period: Optional[float] = None) -> float:
        """Year-fraction accrual factor for a single period.

        For ACT/360: alpha = period (or maturity) in years * 360/360 = period.
        For ACT/365: alpha = period * 365/365 = period (already in year fracs).

        Example: a 3-month deposit with ACT/360: alpha = 0.25.
        """
        t = period if period is not None else self.maturity_years
        if self.day_count == "ACT/365":
            return t  # year fractions already correct
        return t  # ACT/360 -- maturity_years already expressed in year fractions


# ---------------------------------------------------------------------------
# Discount Curve
# ---------------------------------------------------------------------------

class DiscountCurve:
    """Interpolated OIS discount curve.

    Stores calibrated (time, discount-factor) pairs and provides
    interpolated values plus derived quantities (zero rates, forwards,
    par rates).

    Parameters
    ----------
    times : array-like of float
        Year fractions for each knot, starting from 0.0.
    dfs : array-like of float
        Discount factors at each knot, with dfs[0] = 1.0.
    valuation_date : datetime.date
        The T=0 anchor date, e.g. ``datetime.date(2026, 3, 5)``.
    interpolation_method : str
        ``"log_linear"`` (default) or ``"monotone_convex"``.
    """

    def __init__(
        self,
        times: Sequence[float],
        dfs: Sequence[float],
        valuation_date: dt.date,
        interpolation_method: str = "log_linear",
    ) -> None:
        self.times = np.asarray(times, dtype=float)
        self.dfs = np.asarray(dfs, dtype=float)
        self.valuation_date = valuation_date
        self.interpolation_method = interpolation_method

        if interpolation_method == "log_linear":
            log_dfs = np.log(self.dfs)
            self._interp = LogLinearInterpolator(self.times, log_dfs)
            self._interp_mode = "log_df"
        elif interpolation_method == "monotone_convex":
            # Build zero rates (avoid division by zero at t=0)
            zero_rates = np.zeros_like(self.times)
            for i in range(1, len(self.times)):
                zero_rates[i] = -np.log(self.dfs[i]) / self.times[i]
            zero_rates[0] = zero_rates[1] if len(zero_rates) > 1 else 0.0
            self._interp = MonotoneConvexInterpolator(self.times, zero_rates)
            self._interp_mode = "zero_rate"
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

    def df(self, t: float) -> float:
        """Discount factor at year fraction *t*.

        Example: ``curve.df(1.0)`` might return 0.9575 (i.e. $1 in 1 year
        is worth $0.9575 today at ~4.3% rates).
        """
        if t <= 0.0:
            return 1.0
        if self._interp_mode == "log_df":
            return math.exp(self._interp(t))
        else:
            # monotone convex: interp gives zero rate
            r = self._interp(t)
            return math.exp(-r * t)

    def discount(self, t: float) -> float:
        """Alias for df(t) — compatibility with QuantLib-style interfaces."""
        return self.df(t)

    def zero_rate(self, t: float) -> float:
        """Continuously-compounded zero rate at year fraction *t*.

        Defined as ``r(t) = -ln(D(t)) / t``.

        Example: if D(5) = 0.8070, then r(5) = -ln(0.8070)/5 = 0.04286.
        """
        if t <= 1e-12:
            # Limit: use instantaneous rate at the short end
            return self.zero_rate(1e-4)
        return -math.log(self.df(t)) / t

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simply-compounded forward rate between *t1* and *t2*.

        ``F(t1, t2) = (D(t1)/D(t2) - 1) / (t2 - t1)``

        Example: if D(1)=0.9575 and D(2)=0.9160, then
        F(1,2) = (0.9575/0.9160 - 1)/1.0 = 0.04530 (4.53%).
        """
        if t2 <= t1:
            raise ValueError(f"t2={t2} must be > t1={t1}")
        d1 = self.df(t1)
        d2 = self.df(t2)
        return (d1 / d2 - 1.0) / (t2 - t1)

    def instantaneous_forward(self, t: float, dt_bump: float = 1e-4) -> float:
        """Instantaneous forward rate at *t* via finite difference.

        ``f(t) = -d/dt ln D(t) ~ (ln D(t) - ln D(t+dt)) / dt``

        Example: at t=5.0, if D(5.0)=0.8070 and D(5.0001)=0.80696,
        f(5) ~ (ln(0.8070)-ln(0.80696))/0.0001 ~ 0.0428.
        """
        if t < 0.0:
            t = 0.0
        log_d1 = math.log(self.df(t))
        log_d2 = math.log(self.df(t + dt_bump))
        return -(log_d2 - log_d1) / dt_bump

    def par_rate(self, maturity: float, frequency: float = 1.0) -> float:
        """Par swap rate for a given *maturity* and payment *frequency*.

        ``S = (1 - D(T)) / sum(alpha_j * D(t_j))``

        Example: with annual payments and D(1)=0.9575, D(2)=0.9160,
        S = (1 - 0.9160) / (1.0*0.9575 + 1.0*0.9160) = 0.0840/1.8735 = 0.04484.
        """
        n_periods = max(1, int(round(maturity / frequency)))
        schedule = [frequency * (i + 1) for i in range(n_periods)]
        # Adjust last period to exact maturity
        schedule[-1] = maturity

        annuity = sum(frequency * self.df(t) for t in schedule)
        if annuity < 1e-15:
            return 0.0
        return (1.0 - self.df(maturity)) / annuity


# ---------------------------------------------------------------------------
# Bootstrapper
# ---------------------------------------------------------------------------

class CurveBootstrapper:
    """Sequential OIS curve bootstrapper.

    Builds a ``DiscountCurve`` from a list of ``CurveInstrument`` objects,
    solving for discount factors from shortest to longest maturity.

    Parameters
    ----------
    interpolation_method : str
        Passed through to the ``DiscountCurve``.  Default ``"log_linear"``.
    """

    def __init__(self, interpolation_method: str = "log_linear") -> None:
        self.interpolation_method = interpolation_method

    def bootstrap(
        self,
        instruments: List[CurveInstrument],
        valuation_date: dt.date,
    ) -> DiscountCurve:
        """Bootstrap a discount curve from calibration instruments.

        Instruments are sorted by maturity and processed sequentially:

        1. **Deposits**: ``D(t) = 1 / (1 + r * alpha)``
        2. **Swaps**: ``D(t_n) = (1 - S_n * A) / (1 + S_n * alpha_n)``
           where A = sum of (alpha_j * D(t_j)) for all prior payment dates.

        Parameters
        ----------
        instruments : list of CurveInstrument
            Must include at least one deposit for the short end.
        valuation_date : datetime.date
            The T=0 date.

        Returns
        -------
        DiscountCurve
            The calibrated discount curve.
        """
        # Sort by maturity
        instruments = sorted(instruments, key=lambda x: x.maturity_years)

        # Attempt C++ bootstrap if available
        if _USE_CPP and _pk is not None:
            try:
                return self._bootstrap_cpp(instruments, valuation_date)
            except Exception as exc:
                logger.warning("C++ bootstrap failed, falling back to Python: %s", exc)

        return self._bootstrap_python(instruments, valuation_date)

    def _bootstrap_python(
        self,
        instruments: List[CurveInstrument],
        valuation_date: dt.date,
    ) -> DiscountCurve:
        """Pure-Python sequential bootstrap."""
        times = [0.0]
        dfs = [1.0]

        # Track all payment times and their DFs for swap annuity calculation
        known_times: List[float] = [0.0]
        known_dfs: List[float] = [1.0]

        for inst in instruments:
            if inst.type == "deposit":
                # D(t) = 1 / (1 + r * alpha)
                alpha = inst.alpha()
                d = 1.0 / (1.0 + inst.rate * alpha)
                times.append(inst.maturity_years)
                dfs.append(d)
                known_times.append(inst.maturity_years)
                known_dfs.append(d)

            elif inst.type == "swap":
                # Build payment schedule
                n_periods = max(1, int(round(inst.maturity_years / inst.payment_frequency)))
                schedule = [inst.payment_frequency * (i + 1) for i in range(n_periods)]
                schedule[-1] = inst.maturity_years  # ensure exact maturity

                # Compute annuity from already-known discount factors
                # For intermediate payment dates not yet in our curve,
                # use log-linear interpolation on what we have so far
                tmp_curve = DiscountCurve(
                    known_times, known_dfs, valuation_date, self.interpolation_method
                )

                annuity_sum = 0.0
                for j, t_j in enumerate(schedule[:-1]):
                    alpha_j = inst.payment_frequency
                    d_j = tmp_curve.df(t_j)
                    annuity_sum += alpha_j * d_j

                # Solve for D(t_n):
                # D(t_n) = (1 - S * annuity_sum) / (1 + S * alpha_n)
                alpha_n = inst.payment_frequency
                s_n = inst.rate
                d_n = (1.0 - s_n * annuity_sum) / (1.0 + s_n * alpha_n)

                times.append(inst.maturity_years)
                dfs.append(d_n)
                known_times.append(inst.maturity_years)
                known_dfs.append(d_n)
            else:
                raise ValueError(f"Unknown instrument type: {inst.type!r}")

        return DiscountCurve(times, dfs, valuation_date, self.interpolation_method)

    def _bootstrap_cpp(
        self,
        instruments: List[CurveInstrument],
        valuation_date: dt.date,
    ) -> DiscountCurve:
        """Delegate bootstrap to the C++ pricing_kernel extension."""
        maturities = [inst.maturity_years for inst in instruments]
        rates = [inst.rate for inst in instruments]
        types = [inst.type for inst in instruments]
        freqs = [inst.payment_frequency for inst in instruments]

        result = _pk.bootstrap_ois(maturities, rates, types, freqs)  # type: ignore[union-attr]
        times = [0.0] + list(result["times"])
        dfs_out = [1.0] + list(result["dfs"])
        return DiscountCurve(times, dfs_out, valuation_date, self.interpolation_method)

    def validate(
        self,
        curve: DiscountCurve,
        instruments: List[CurveInstrument],
    ) -> pd.DataFrame:
        """Re-price each calibration instrument and report errors.

        Returns a DataFrame with columns:
        ``type, maturity, market_rate, model_rate, error_bps``.

        A well-calibrated curve should show errors < 0.01 bps for all
        instruments.
        """
        rows = []
        for inst in instruments:
            if inst.type == "deposit":
                # Implied rate from DF: r = (1/D - 1) / alpha
                alpha = inst.alpha()
                d = curve.df(inst.maturity_years)
                model_rate = (1.0 / d - 1.0) / alpha if alpha > 0 else 0.0
            elif inst.type == "swap":
                model_rate = curve.par_rate(inst.maturity_years, inst.payment_frequency)
            else:
                model_rate = float("nan")

            error_bps = (model_rate - inst.rate) * 10_000
            rows.append(
                {
                    "type": inst.type,
                    "maturity": inst.maturity_years,
                    "market_rate": inst.rate,
                    "model_rate": round(model_rate, 8),
                    "error_bps": round(error_bps, 4),
                }
            )

        return pd.DataFrame(rows)
