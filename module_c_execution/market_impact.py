"""
Almgren-Chriss market impact model calibrated for CME FX futures.

Estimates permanent impact, temporary impact, and timing risk for executing
block orders across the major CME FX futures contracts (6E, 6B, 6J, 6A).
All costs are expressed in pips, the standard unit for FX markets.

References
----------
Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions."
    Journal of Risk, 3(2), 5-39.
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# CME FX futures contract specifications
# ---------------------------------------------------------------------------
FX_FUTURES: Dict[str, dict] = {
    "6E": {
        "name": "EUR/USD",
        "contract_size": 125_000,
        "tick_size": 0.00005,
        "tick_value": 6.25,
        "avg_daily_volume": 250_000,
        "daily_vol_pips": 55,
        "margin": 2_600,
    },
    "6B": {
        "name": "GBP/USD",
        "contract_size": 62_500,
        "tick_size": 0.0001,
        "tick_value": 6.25,
        "avg_daily_volume": 120_000,
        "daily_vol_pips": 75,
        "margin": 2_400,
    },
    "6J": {
        "name": "JPY/USD",
        "contract_size": 12_500_000,
        "tick_size": 0.0000005,
        "tick_value": 6.25,
        "avg_daily_volume": 180_000,
        "daily_vol_pips": 60,
        "margin": 3_200,
    },
    "6A": {
        "name": "AUD/USD",
        "contract_size": 100_000,
        "tick_size": 0.0001,
        "tick_value": 10.00,
        "avg_daily_volume": 95_000,
        "daily_vol_pips": 85,
        "margin": 1_800,
    },
}

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
ImpactResult = namedtuple(
    "ImpactResult",
    [
        "permanent_cost_pips",
        "temporary_cost_pips",
        "total_cost_pips",
        "optimal_horizon_minutes",
        "participation_rate",
    ],
)


# ---------------------------------------------------------------------------
# Almgren-Chriss model
# ---------------------------------------------------------------------------
class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution model for CME FX futures.

    Parameters
    ----------
    gamma_frac : float
        Permanent impact coefficient as a fraction of daily volume.
        Default 0.05 means moving 100% of daily volume would cause
        permanent impact of 5% of daily volatility (e.g. 0.05 * 55 = 2.75 pips
        for 6E when Q == V_daily).
    eta_frac : float
        Temporary impact coefficient as a fraction of per-minute volume.
        Default 0.10 means each execution slice's temporary impact scales as
        10% of daily volatility at full participation.
    lambda_risk : float
        Risk-aversion parameter that penalises variance of execution cost.
        Higher values produce faster (more aggressive) execution.
        Default 1e-6.
    """

    # CME Globex FX futures trade ~23 h/day; we use the liquid 6.5 h window
    TRADING_DAY_MINUTES: int = 390  # 06:00-12:30 CT core liquidity window

    def __init__(
        self,
        gamma_frac: float = 0.05,
        eta_frac: float = 0.10,
        lambda_risk: float = 1e-6,
    ) -> None:
        self.gamma_frac = gamma_frac
        self.eta_frac = eta_frac
        self.lambda_risk = lambda_risk

    # -----------------------------------------------------------------
    # Core cost computation
    # -----------------------------------------------------------------
    def total_cost(
        self,
        n_contracts: int,
        daily_volume: int,
        daily_vol_pips: float,
        horizon_minutes: float,
    ) -> Dict[str, float]:
        """Compute execution cost breakdown for a given horizon.

        Parameters
        ----------
        n_contracts : int
            Number of contracts to execute (e.g. 50).
        daily_volume : int
            Average daily volume in contracts (e.g. 250_000 for 6E).
        daily_vol_pips : float
            Annualised intraday volatility in pips (e.g. 55 for 6E).
        horizon_minutes : float
            Planned execution window in minutes (e.g. 30.0).

        Returns
        -------
        dict
            Keys: permanent_cost_pips, temporary_cost_pips, total_cost_pips,
            timing_risk_pips.

        Examples
        --------
        >>> model = AlmgrenChrissModel()
        >>> result = model.total_cost(100, 250_000, 55, 30.0)
        >>> # permanent ~ 0.05 * (100/250000) * 55 ≈ 0.0011 pips
        >>> # temporary ~ 0.10 * (100/(30*641)) * 55 ≈ 0.0286 pips
        """
        Q = float(n_contracts)
        V = float(daily_volume)
        sigma = float(daily_vol_pips)
        T = float(horizon_minutes)

        # Volume per minute during trading day
        V_per_min = V / self.TRADING_DAY_MINUTES  # e.g. 250000/390 ≈ 641

        # Permanent impact: market moves against you proportional to order size
        permanent = self.gamma_frac * (Q / V) * sigma

        # Temporary impact: price concession you pay for immediacy
        if T > 0 and V_per_min > 0:
            temporary = self.eta_frac * (Q / (T * V_per_min)) * sigma
        else:
            temporary = 0.0

        # Timing risk: price can drift while you wait to finish
        T_day = float(self.TRADING_DAY_MINUTES)
        timing_risk = 0.5 * sigma * math.sqrt(T / T_day) if T > 0 else 0.0

        total = permanent + temporary

        return {
            "permanent_cost_pips": round(permanent, 6),
            "temporary_cost_pips": round(temporary, 6),
            "total_cost_pips": round(total, 6),
            "timing_risk_pips": round(timing_risk, 6),
        }

    # -----------------------------------------------------------------
    # Optimal horizon
    # -----------------------------------------------------------------
    def optimal_horizon(
        self,
        n_contracts: int,
        daily_volume: int,
        daily_vol_pips: float,
    ) -> float:
        """Find the horizon that minimises total cost + timing risk.

        Uses the Almgren-Chriss closed-form result:
            kappa = sqrt(lambda * sigma² / eta)
            T*    = (1 / kappa) * arccosh(1 + kappa * Q / V_daily)

        Parameters
        ----------
        n_contracts : int
            Order size in contracts.
        daily_volume : int
            Average daily volume in contracts.
        daily_vol_pips : float
            Daily volatility in pips.

        Returns
        -------
        float
            Optimal execution horizon in minutes.

        Examples
        --------
        >>> model = AlmgrenChrissModel()
        >>> model.optimal_horizon(100, 250_000, 55)
        # Returns something like 4.5 minutes for a small 6E order
        """
        Q = float(n_contracts)
        V = float(daily_volume)
        sigma = float(daily_vol_pips)

        # eta in absolute terms (pips per contract per minute of trading)
        V_per_min = V / self.TRADING_DAY_MINUTES
        eta_abs = self.eta_frac * sigma / V_per_min if V_per_min > 0 else 1e-12

        # kappa: urgency parameter
        lam = self.lambda_risk
        kappa = math.sqrt(lam * sigma**2 / eta_abs) if eta_abs > 0 else 1e-12

        # Optimal horizon via arccosh
        arg = 1.0 + kappa * Q / V
        if arg < 1.0:
            arg = 1.0  # clamp for numerical safety
        T_star = (1.0 / kappa) * math.acosh(arg) if kappa > 0 else 60.0

        # Clamp to [1 minute, full trading day]
        T_star = max(1.0, min(T_star, float(self.TRADING_DAY_MINUTES)))

        return round(T_star, 2)

    # -----------------------------------------------------------------
    # Convenience: cost for a named futures ticker
    # -----------------------------------------------------------------
    def cost_for_futures(
        self,
        ticker: str,
        n_contracts: int,
    ) -> ImpactResult:
        """Compute optimal execution cost for a specific CME FX future.

        Parameters
        ----------
        ticker : str
            CME futures ticker, e.g. ``"6E"``, ``"6B"``, ``"6J"``, ``"6A"``.
        n_contracts : int
            Number of contracts to execute.

        Returns
        -------
        ImpactResult
            Named tuple with permanent_cost_pips, temporary_cost_pips,
            total_cost_pips, optimal_horizon_minutes, participation_rate.

        Raises
        ------
        KeyError
            If *ticker* is not found in ``FX_FUTURES``.

        Examples
        --------
        >>> model = AlmgrenChrissModel()
        >>> r = model.cost_for_futures("6E", 200)
        >>> r.total_cost_pips   # permanent + temporary at optimal horizon
        """
        spec = FX_FUTURES[ticker]
        vol = spec["avg_daily_volume"]
        sigma = spec["daily_vol_pips"]

        opt_T = self.optimal_horizon(n_contracts, vol, sigma)
        costs = self.total_cost(n_contracts, vol, sigma, opt_T)

        # Participation rate = fraction of per-minute volume consumed
        V_per_min = vol / self.TRADING_DAY_MINUTES
        participation = (n_contracts / opt_T) / V_per_min if opt_T > 0 else 0.0

        return ImpactResult(
            permanent_cost_pips=costs["permanent_cost_pips"],
            temporary_cost_pips=costs["temporary_cost_pips"],
            total_cost_pips=costs["total_cost_pips"],
            optimal_horizon_minutes=opt_T,
            participation_rate=round(participation, 6),
        )

    # -----------------------------------------------------------------
    # Impact curve for plotting
    # -----------------------------------------------------------------
    def impact_curve(
        self,
        n_contracts: int,
        daily_volume: int,
        daily_vol_pips: float,
    ) -> Dict[str, np.ndarray]:
        """Generate cost arrays across a range of horizons for plotting.

        Sweeps horizon from 1 minute to the full trading day (390 min) and
        returns arrays of permanent, temporary, total, and timing-risk costs.

        Parameters
        ----------
        n_contracts : int
            Order size in contracts.
        daily_volume : int
            Average daily volume in contracts.
        daily_vol_pips : float
            Daily volatility in pips.

        Returns
        -------
        dict
            Keys: horizons, permanent, temporary, total, timing_risk — each a
            numpy array of length 200.

        Examples
        --------
        >>> model = AlmgrenChrissModel()
        >>> curves = model.impact_curve(500, 250_000, 55)
        >>> curves["horizons"][:3]  # array([1.0, 2.97, 4.94, ...])
        """
        horizons = np.linspace(1.0, float(self.TRADING_DAY_MINUTES), 200)
        permanent = np.zeros_like(horizons)
        temporary = np.zeros_like(horizons)
        total = np.zeros_like(horizons)
        timing_risk = np.zeros_like(horizons)

        for i, h in enumerate(horizons):
            c = self.total_cost(n_contracts, daily_volume, daily_vol_pips, h)
            permanent[i] = c["permanent_cost_pips"]
            temporary[i] = c["temporary_cost_pips"]
            total[i] = c["total_cost_pips"]
            timing_risk[i] = c["timing_risk_pips"]

        return {
            "horizons": horizons,
            "permanent": permanent,
            "temporary": temporary,
            "total": total,
            "timing_risk": timing_risk,
        }
