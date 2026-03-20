"""
FX Forward Pricing via Covered Interest Parity (CIP).

The no-arbitrage forward rate is:

    F(T) = S * D_foreign(T) / D_domestic(T)

For example, EUR/USD with S=1.0832, D_EUR(1Y)=0.9714, D_USD(1Y)=0.9575:
    F(1Y) = 1.0832 * 0.9714 / 0.9575 = 1.0985
    Forward points = 1.0985 - 1.0832 = 0.0153 (153 pips)

In practice, cross-currency basis means the actual traded forwards deviate
from this by a few basis points (the CIP basis).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module_a_curves.curve_bootstrapper import DiscountCurve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard Tenors (year fractions)
# ---------------------------------------------------------------------------
STANDARD_TENORS: Dict[str, float] = {
    "1M": 1.0 / 12,
    "3M": 3.0 / 12,
    "6M": 6.0 / 12,
    "1Y": 1.0,
    "2Y": 2.0,
    "5Y": 5.0,
}

# ---------------------------------------------------------------------------
# Realistic Cross-Currency Basis (bps), typical 2024-2026 levels
# Negative basis means foreign borrowers pay a premium to swap into USD.
# ---------------------------------------------------------------------------
_REALISTIC_BASIS_BPS: Dict[str, Dict[str, float]] = {
    "EUR/USD": {
        "1M": -15.0, "3M": -17.0, "6M": -19.0,
        "1Y": -21.0, "2Y": -23.0, "5Y": -25.0,
    },
    "GBP/USD": {
        "1M": -5.0, "3M": -7.0, "6M": -9.0,
        "1Y": -11.0, "2Y": -13.0, "5Y": -15.0,
    },
    "JPY/USD": {
        "1M": -30.0, "3M": -34.0, "6M": -38.0,
        "1Y": -42.0, "2Y": -46.0, "5Y": -50.0,
    },
    "AUD/USD": {
        "1M": -8.0, "3M": -10.0, "6M": -13.0,
        "1Y": -16.0, "2Y": -19.0, "5Y": -22.0,
    },
}


class FXForwardCurve:
    """FX forward curve derived from CIP.

    Parameters
    ----------
    spot_rate : float
        Current spot FX rate, e.g. 1.0832 for EUR/USD.
    domestic_curve : DiscountCurve
        USD OIS discount curve (the domestic / numeraire currency).
    foreign_curve : DiscountCurve
        Foreign-currency OIS discount curve (e.g. EUR curve).
    pair_name : str
        Currency pair label, e.g. ``"EUR/USD"``.
    """

    def __init__(
        self,
        spot_rate: float,
        domestic_curve: DiscountCurve,
        foreign_curve: DiscountCurve,
        pair_name: str = "EUR/USD",
    ) -> None:
        self.spot_rate = spot_rate
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.pair_name = pair_name

    def forward(self, T: float) -> float:
        """CIP-implied FX forward rate at maturity *T* (in years).

        Formula: ``F(T) = S * D_foreign(T) / D_domestic(T)``

        Example (EUR/USD, T=1):
            S=1.0832, D_EUR(1)=0.9714, D_USD(1)=0.9575
            F = 1.0832 * 0.9714 / 0.9575 = 1.0985
        """
        if T <= 0.0:
            return self.spot_rate
        d_dom = self.domestic_curve.df(T)
        d_for = self.foreign_curve.df(T)
        return self.spot_rate * d_for / d_dom

    def forward_points(self, T: float) -> float:
        """Forward points at maturity *T* in natural units (not pips).

        ``forward_points = F(T) - S``

        For EUR/USD: if F(1Y)=1.0985, S=1.0832, points = +0.0153 (= +153 pips).
        For JPY/USD: points are much smaller given the spot ~0.00667.
        """
        return self.forward(T) - self.spot_rate

    def forward_points_pips(self, T: float) -> float:
        """Forward points in pips (1 pip = 0.0001 for most pairs).

        Example: EUR/USD forward points of 0.0153 = 153.0 pips.
        """
        return self.forward_points(T) / 0.0001

    def implied_basis(self, T: float) -> float:
        """CIP deviation (implied basis) at maturity *T*, in basis points.

        Computed as the spread *b* such that:
            F(T) = S * D_foreign(T) / D_domestic_adjusted(T)
        where D_domestic_adjusted uses (r_dom + b).

        In a frictionless market this would be zero.  In practice,
        EUR/USD shows -15 to -25 bps, JPY/USD shows -30 to -50 bps.

        Here we compute:
            b = (1/T) * [ln(D_dom(T)) - ln(D_for(T)) - ln(S) + ln(F_market(T))]
        Since we only have the CIP-implied forward (no market forward),
        the basis from this curve alone is identically zero.  For
        realistic basis values, see ``CrossCurrencyBasis``.
        """
        if T <= 1e-12:
            return 0.0
        # Theoretical basis from our own curves (should be ~0)
        f = self.forward(T)
        import math
        implied_yield_dom = -math.log(self.domestic_curve.df(T)) / T
        implied_yield_for = -math.log(self.foreign_curve.df(T)) / T
        cip_forward = self.spot_rate * math.exp((implied_yield_dom - implied_yield_for) * T)
        deviation = (f / cip_forward - 1.0) / T if T > 0 else 0.0
        return deviation * 10_000  # convert to bps

    def plot_forward_curve(
        self,
        ax: Optional[object] = None,
        max_maturity: float = 5.0,
        n_points: int = 100,
    ):
        """Plot the FX forward curve from spot out to *max_maturity*.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            If *None*, creates a new figure.
        max_maturity : float
            Maximum maturity in years (default 5).
        n_points : int
            Number of evaluation points (default 100).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        tenors = np.linspace(0.01, max_maturity, n_points)
        forwards = [self.forward(t) for t in tenors]
        points_pips = [self.forward_points_pips(t) for t in tenors]

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{self.pair_name} Forward Curve (S = {self.spot_rate:.4f})")
        else:
            # If a single axes is provided, only plot the forward rate
            ax.plot(tenors, forwards, "b-", linewidth=1.5)
            ax.set_xlabel("Maturity (years)")
            ax.set_ylabel("Forward Rate")
            ax.set_title(f"{self.pair_name} Forward Curve")
            ax.axhline(self.spot_rate, color="gray", linestyle="--", alpha=0.5, label="Spot")
            ax.legend()
            ax.grid(True, alpha=0.3)
            return ax

        # Left panel: forward rate
        ax[0].plot(tenors, forwards, "b-", linewidth=1.5)
        ax[0].axhline(self.spot_rate, color="gray", linestyle="--", alpha=0.5, label="Spot")
        ax[0].set_xlabel("Maturity (years)")
        ax[0].set_ylabel("Forward Rate")
        ax[0].set_title("Outright Forward")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Right panel: forward points in pips
        ax[1].plot(tenors, points_pips, "r-", linewidth=1.5)
        ax[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax[1].set_xlabel("Maturity (years)")
        ax[1].set_ylabel("Forward Points (pips)")
        ax[1].set_title("Forward Points")
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return ax


class CrossCurrencyBasis:
    """Cross-currency basis for G4 pairs at standard tenors.

    CIP deviations represent the cost of synthetically borrowing USD via
    FX swaps vs. borrowing directly.  Typical 2024-2026 values:

    - EUR/USD: -15 to -25 bps (ECB excess liquidity drives negative basis)
    - GBP/USD: -5 to -15 bps (modest basis)
    - JPY/USD: -30 to -50 bps (wide basis due to BOJ yield-curve control legacy)
    - AUD/USD: -8 to -22 bps (commodity-currency basis)

    These are hardcoded realistic values.  In production, one would
    calibrate from observed FX swap points vs. OIS differentials.
    """

    def __init__(self) -> None:
        self._basis_data = _REALISTIC_BASIS_BPS

    @property
    def pairs(self) -> List[str]:
        """Available currency pairs."""
        return list(self._basis_data.keys())

    def basis_curve(self, pair: str) -> pd.DataFrame:
        """Return the cross-currency basis curve for *pair*.

        Parameters
        ----------
        pair : str
            Currency pair, e.g. ``"EUR/USD"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``tenor`` (str), ``maturity_years`` (float),
            ``basis_bps`` (float).

        Example output for EUR/USD::

            tenor  maturity_years  basis_bps
            1M     0.083333        -15.0
            3M     0.250000        -17.0
            6M     0.500000        -19.0
            1Y     1.000000        -21.0
            2Y     2.000000        -23.0
            5Y     5.000000        -25.0
        """
        if pair not in self._basis_data:
            raise ValueError(
                f"Unknown pair {pair!r}. Available: {self.pairs}"
            )
        basis = self._basis_data[pair]
        rows = []
        for tenor_label, t_years in STANDARD_TENORS.items():
            rows.append(
                {
                    "tenor": tenor_label,
                    "maturity_years": round(t_years, 6),
                    "basis_bps": basis.get(tenor_label, 0.0),
                }
            )
        return pd.DataFrame(rows)

    def basis_at_tenor(self, pair: str, tenor: str) -> float:
        """Return the basis in bps for a specific *pair* and *tenor*.

        Example: ``basis_at_tenor("EUR/USD", "1Y")`` returns ``-21.0``.
        """
        if pair not in self._basis_data:
            raise ValueError(f"Unknown pair {pair!r}. Available: {self.pairs}")
        return self._basis_data[pair].get(tenor, 0.0)

    def summary(self) -> pd.DataFrame:
        """Return a summary table of basis values for all pairs and tenors.

        Columns are tenor labels, rows are currency pairs, values are bps.

        Example::

                    1M    3M    6M    1Y    2Y    5Y
            EUR/USD -15.0 -17.0 -19.0 -21.0 -23.0 -25.0
            GBP/USD  -5.0  -7.0  -9.0 -11.0 -13.0 -15.0
            ...
        """
        rows = {}
        for pair in self.pairs:
            rows[pair] = {
                tenor: self._basis_data[pair].get(tenor, 0.0)
                for tenor in STANDARD_TENORS
            }
        return pd.DataFrame(rows).T
