"""
Execution scheduling strategies for CME FX futures.

Provides three standard execution algorithms — TWAP, VWAP, and Adaptive
(sinh-urgency) — each producing a sequence of ``ExecutionSlice`` objects
that describe how to split a block order across time.

The VWAP profile is calibrated to the empirical intraday volume pattern of
CME FX futures, which shows a distinctive U-shape driven by overlapping
European and North American sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from module_c_execution.vwap_calibration import calibrate_vwap_profile_24h

# ---------------------------------------------------------------------------
# Slice container
# ---------------------------------------------------------------------------

@dataclass
class ExecutionSlice:
    """Single execution slice within a scheduled trajectory.

    Attributes
    ----------
    time_fraction : float
        Position in [0, 1] representing when this slice fires.
        For example 0.0 is the start, 0.5 is the midpoint, 1.0 is the end.
    quantity : float
        Number of contracts to execute in this slice (e.g. 7.69).
    cumulative_quantity : float
        Running total of contracts executed up to and including this slice.
    price_adjustment : float
        Expected price concession in pips for this slice (informational).
    """

    time_fraction: float
    quantity: float
    cumulative_quantity: float
    price_adjustment: float = 0.0


# ---------------------------------------------------------------------------
# CME FX intraday volume profile (UTC hours 0-23)
# ---------------------------------------------------------------------------
# Legacy hardcoded fallback used only if MBP-10 calibration produces no data.
# The live profile is computed from cached CME MBP-10 deep-book parquets via
# ``vwap_calibration.calibrate_vwap_profile_24h`` (see that module for the
# 13:00-17:00 UTC real-data window and merge logic).
_LEGACY_FX_VOLUME_PROFILE_24H = np.array(
    [
        # 00  01  02  03  04  05  06  07  08  09  10  11
        0.03, 0.02, 0.02, 0.03, 0.04, 0.05, 0.10, 0.60, 0.85, 0.75, 0.65, 0.60,
        # 12  13  14  15  16  17  18  19  20  21  22  23
        0.70, 0.90, 1.00, 0.95, 0.80, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.03,
    ],
    dtype=np.float64,
)
_LEGACY_FX_VOLUME_PROFILE_24H = (
    _LEGACY_FX_VOLUME_PROFILE_24H / _LEGACY_FX_VOLUME_PROFILE_24H.sum()
)


def _fx_volume_profile_24h() -> np.ndarray:
    """Return a 24-element array of relative volume weights by UTC hour.

    The profile is calibrated from cached CME MBP-10 deep-book parquets
    (13:00-17:00 UTC real-data window), with the remainder of the day
    filled in from a U-shaped expert curve and the whole curve normalised
    to integrate to 1.0.  If no cached data is available, the legacy
    hardcoded profile is returned as a fallback.

    Returns
    -------
    np.ndarray
        Shape (24,), normalised so that the values sum to 1.0.
    """
    try:
        profile = calibrate_vwap_profile_24h()
    except Exception:
        profile = None
    if profile is None or len(profile) != 24 or not np.isfinite(profile).all() or profile.sum() <= 0:
        return _LEGACY_FX_VOLUME_PROFILE_24H.copy()
    return profile / profile.sum()


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------
class TWAPScheduler:
    """Time-Weighted Average Price scheduler.

    Splits the total quantity into *n_slices* equal-sized slices evenly
    spaced in time.  This is the simplest benchmark strategy that ignores
    intraday volume patterns.

    Parameters
    ----------
    n_slices : int
        Number of execution slices.  Default is 13, which gives
        ~30-minute intervals across a 6.5-hour CME FX session.

    Examples
    --------
    >>> sched = TWAPScheduler(n_slices=5)
    >>> slices = sched.schedule(100)
    >>> [s.quantity for s in slices]
    [20.0, 20.0, 20.0, 20.0, 20.0]
    """

    def __init__(self, n_slices: int = 13) -> None:
        self.n_slices = n_slices

    def schedule(self, total_quantity: int) -> List[ExecutionSlice]:
        """Generate a uniform execution schedule.

        Parameters
        ----------
        total_quantity : int
            Total number of contracts to execute.

        Returns
        -------
        list[ExecutionSlice]
            List of *n_slices* slices with equal quantity.
        """
        q_per_slice = total_quantity / self.n_slices
        slices: List[ExecutionSlice] = []
        cumulative = 0.0

        for i in range(self.n_slices):
            t_frac = i / (self.n_slices - 1) if self.n_slices > 1 else 0.0
            cumulative += q_per_slice
            slices.append(
                ExecutionSlice(
                    time_fraction=round(t_frac, 6),
                    quantity=round(q_per_slice, 4),
                    cumulative_quantity=round(cumulative, 4),
                )
            )

        return slices


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------
class VWAPScheduler:
    """Volume-Weighted Average Price scheduler for CME FX futures.

    Allocates quantity proportional to the empirical intraday volume profile
    of CME FX futures.  The profile is U-shaped with peaks at European open
    and the London/NY overlap.

    The 24-hour volume profile is sampled at *n_slices* evenly spaced
    points, and each slice receives quantity proportional to the volume
    weight at that time.

    Parameters
    ----------
    n_slices : int
        Number of execution slices.  Default 13.

    Examples
    --------
    >>> sched = VWAPScheduler(n_slices=5)
    >>> slices = sched.schedule(100)
    >>> # First slice (Asian) gets fewer contracts than midday (NY overlap)
    >>> slices[0].quantity < slices[2].quantity
    True
    """

    def __init__(self, n_slices: int = 13) -> None:
        self.n_slices = n_slices
        self._profile_24h = _fx_volume_profile_24h()

    def schedule(self, total_quantity: int) -> List[ExecutionSlice]:
        """Generate a volume-weighted execution schedule.

        Parameters
        ----------
        total_quantity : int
            Total number of contracts to execute.

        Returns
        -------
        list[ExecutionSlice]
            List of *n_slices* slices weighted by CME FX volume profile.
        """
        # Map each slice index to a UTC hour within the core session
        # Core session: 07:00 - 17:00 UTC (10 hours of meaningful liquidity)
        core_start_hour = 7
        core_end_hour = 17
        core_hours = core_end_hour - core_start_hour  # 10 hours

        weights = np.zeros(self.n_slices)
        for i in range(self.n_slices):
            # Map slice to a fractional hour in [core_start, core_end]
            t_frac = i / (self.n_slices - 1) if self.n_slices > 1 else 0.0
            hour_float = core_start_hour + t_frac * core_hours
            hour_idx = int(np.clip(hour_float, 0, 23))
            weights[i] = self._profile_24h[hour_idx]

        # Normalise so weights sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum

        slices: List[ExecutionSlice] = []
        cumulative = 0.0

        for i in range(self.n_slices):
            t_frac = i / (self.n_slices - 1) if self.n_slices > 1 else 0.0
            qty = total_quantity * weights[i]
            cumulative += qty
            slices.append(
                ExecutionSlice(
                    time_fraction=round(t_frac, 6),
                    quantity=round(qty, 4),
                    cumulative_quantity=round(cumulative, 4),
                )
            )

        return slices


# ---------------------------------------------------------------------------
# Adaptive (sinh-urgency)
# ---------------------------------------------------------------------------
class AdaptiveScheduler:
    """Adaptive execution scheduler using sinh urgency trajectory.

    Follows the Almgren-Chriss optimal trajectory where the execution rate
    is proportional to ``sinh(kappa * t)`` — front-loading execution when
    urgency (kappa) is high.

    Parameters
    ----------
    kappa : float
        Urgency parameter.  Higher values concentrate execution at the start.

        - ``kappa = 0``: equivalent to TWAP (uniform).
        - ``kappa = 1.5`` (default): moderate front-loading.
        - ``kappa = 3.0``: aggressive front-loading.
    n_slices : int
        Number of execution slices.  Default 13.

    Examples
    --------
    >>> sched = AdaptiveScheduler(kappa=2.0, n_slices=5)
    >>> slices = sched.schedule(100)
    >>> # Front-loaded: first slice > last slice
    >>> slices[0].quantity > slices[-1].quantity
    True
    """

    def __init__(self, kappa: float = 1.5, n_slices: int = 13) -> None:
        self.kappa = kappa
        self.n_slices = n_slices

    def schedule(self, total_quantity: int) -> List[ExecutionSlice]:
        """Generate a sinh-urgency weighted execution schedule.

        The incremental quantity for slice *i* is:

            q_i = Q * sinh(kappa * t_i) / sum_j(sinh(kappa * t_j))

        where ``t_i`` ranges from 1/n to 1 (we skip t=0 because sinh(0)=0).

        Parameters
        ----------
        total_quantity : int
            Total number of contracts to execute.

        Returns
        -------
        list[ExecutionSlice]
            List of *n_slices* slices with sinh-weighted quantities.
        """
        # Time points: evenly spaced in (0, 1], skipping 0 where sinh is 0
        t_points = np.linspace(1.0 / self.n_slices, 1.0, self.n_slices)

        if abs(self.kappa) < 1e-12:
            # kappa ≈ 0 → uniform (TWAP)
            weights = np.ones(self.n_slices)
        else:
            weights = np.sinh(self.kappa * (1.0 - t_points + t_points[0]))

        # Normalise
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum

        slices: List[ExecutionSlice] = []
        cumulative = 0.0

        for i in range(self.n_slices):
            t_frac = (i / (self.n_slices - 1)) if self.n_slices > 1 else 0.0
            qty = total_quantity * weights[i]
            cumulative += qty
            slices.append(
                ExecutionSlice(
                    time_fraction=round(t_frac, 6),
                    quantity=round(qty, 4),
                    cumulative_quantity=round(cumulative, 4),
                )
            )

        return slices


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------
def compare_strategies(
    n_contracts: int,
    kappa: float = 1.5,
) -> Dict[str, List[ExecutionSlice]]:
    """Run all three schedulers on the same order and return results.

    Useful for side-by-side comparison of TWAP, VWAP, and Adaptive
    execution trajectories.

    Parameters
    ----------
    n_contracts : int
        Total number of contracts to execute (e.g. 200).
    kappa : float
        Urgency parameter for the AdaptiveScheduler.  Default 1.5.

    Returns
    -------
    dict[str, list[ExecutionSlice]]
        Mapping from strategy name to its schedule.

    Examples
    --------
    >>> results = compare_strategies(100, kappa=2.0)
    >>> list(results.keys())
    ['TWAP', 'VWAP', 'Adaptive']
    >>> len(results['TWAP'])  # 13 slices by default
    13
    """
    return {
        "TWAP": TWAPScheduler().schedule(n_contracts),
        "VWAP": VWAPScheduler().schedule(n_contracts),
        "Adaptive": AdaptiveScheduler(kappa=kappa).schedule(n_contracts),
    }
