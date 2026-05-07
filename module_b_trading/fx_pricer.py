"""
FX Forward/Swap Pricing Engine
==============================

Implements covered interest rate parity (CIP) pricing for FX forwards and swaps,
plus first- and second-order risk sensitivities (delta, gamma, theta).

Forward price:  F(T) = S * D_f(T) / D_d(T)
where S = spot, D_f = foreign discount factor, D_d = domestic discount factor.

Example: EUR/USD spot = 1.0850, EUR 1Y rate = 3.50%, USD 1Y rate = 5.25%
  D_f(1) = exp(-0.035) ~ 0.9656,  D_d(1) = exp(-0.0525) ~ 0.9489
  F(1) = 1.0850 * 0.9656 / 0.9489 ~ 1.1041  (EUR trades at a forward premium vs USD)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FXSwapSpec:
    """Specification for an FX swap trade.

    Attributes:
        notional: Face amount in base currency units (e.g. 10_000_000 EUR).
        maturity_years: Time to maturity in years (e.g. 0.25 for 3 months).
        agreed_forward: The contractual forward rate locked at inception.
        pair: Currency pair string like 'EUR/USD'.
        direction: 'buy_base' (buy EUR, sell USD at maturity) or 'sell_base'.
    """
    notional: float
    maturity_years: float
    agreed_forward: float
    pair: str
    direction: str  # "buy_base" or "sell_base"

    def sign(self) -> int:
        """Return +1 for buy_base, -1 for sell_base."""
        return 1 if self.direction == "buy_base" else -1


class FXPricer:
    """FX forward and swap pricing via covered interest parity.

    Parameters
    ----------
    domestic_curve : callable or object
        Must support ``discount(T)`` returning the domestic discount factor.
        Example: domestic_curve.discount(1.0) -> 0.9489 for USD at 5.25%.
    foreign_curve : callable or object
        Must support ``discount(T)`` returning the foreign discount factor.
        Example: foreign_curve.discount(1.0) -> 0.9656 for EUR at 3.50%.
    spot_rate : float
        Current spot FX rate (domestic per foreign), e.g. 1.0850 for EUR/USD.
    pair_name : str
        Currency pair identifier, e.g. 'EUR/USD'.
    """

    def __init__(self, domestic_curve, foreign_curve, spot_rate: float, pair_name: str,
                 market_curve=None):
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.spot_rate = spot_rate
        self.pair_name = pair_name
        # Optional: a `market_forward.MarketForwardCurve` carrying CME-quoted
        # forwards. When supplied, `fx_swap_npv` marks against the exchange
        # forward instead of the CIP-implied one. Greeks keep using CIP
        # because they need a smooth analytical function.
        self.market_curve = market_curve

    # ------------------------------------------------------------------
    # Core pricing
    # ------------------------------------------------------------------

    def forward_price(self, T: float) -> float:
        """Compute the CIP-implied forward rate for maturity T years.

        F(T) = S * D_f(T) / D_d(T)

        Example (EUR/USD, T=1):
            S = 1.0850, D_f = 0.9656, D_d = 0.9489
            F = 1.0850 * 0.9656 / 0.9489 = 1.1041

        Parameters
        ----------
        T : float
            Time to maturity in years.

        Returns
        -------
        float
            The forward exchange rate.
        """
        d_foreign = self._discount_foreign(T)
        d_domestic = self._discount_domestic(T)
        if d_domestic == 0:
            raise ValueError("Domestic discount factor is zero at T={:.4f}".format(T))
        return self.spot_rate * d_foreign / d_domestic

    def fx_swap_npv(self, notional: float, maturity: float,
                    forward_rate_agreed: float) -> float:
        """NPV of an FX swap in domestic currency terms.

        NPV = notional * (F_market - F_agreed) * D_d(T)

        A positive NPV means the swap is in-the-money for a base-currency buyer.

        Example:
            notional = 10MM EUR, maturity = 1Y
            F_market = 1.1041, F_agreed = 1.1000, D_d(1) = 0.9489
            NPV = 10e6 * (1.1041 - 1.1000) * 0.9489 = +38,905 USD

        Parameters
        ----------
        notional : float
            Notional in base (foreign) currency.
        maturity : float
            Years to maturity.
        forward_rate_agreed : float
            The locked-in forward rate from the swap contract.

        Returns
        -------
        float
            NPV in domestic currency.
        """
        if self.market_curve is not None:
            f_market, _src = self.market_curve.f_market(maturity)
        else:
            f_market = self.forward_price(maturity)
        d_domestic = self._discount_domestic(maturity)
        return notional * (f_market - forward_rate_agreed) * d_domestic

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(self, notional: float, maturity: float) -> float:
        """Spot delta: change in NPV per 1% move in spot (dV/dS * 0.01 * S).

        For a vanilla FX forward, dV/dS = notional * D_f(T) / D_d(T) * D_d(T)
                                         = notional * D_f(T).
        We report the P&L impact of a 1% spot move in domestic terms.

        Example:
            notional = 10MM EUR, D_f(1) = 0.9656
            delta_01 = 10e6 * 0.9656 * 0.01 = 96,560 USD per 1% EUR/USD move

        Parameters
        ----------
        notional : float
            Notional in base currency.
        maturity : float
            Years to maturity.

        Returns
        -------
        float
            Dollar P&L per 1% spot move (domestic terms).
        """
        d_foreign = self._discount_foreign(maturity)
        # dV/dS for the forward leg = notional * D_f(T) (present-valued at domestic rate)
        # We want the P&L for a 1% spot move:
        dv_ds = notional * d_foreign
        return dv_ds * 0.01 * self.spot_rate

    def cross_gamma(self, notional: float, maturity: float) -> float:
        """Second-order spot sensitivity d2V/dS2, computed numerically.

        We bump spot by +/- 0.5% and take the second difference.

        Example:
            For a linear forward, gamma ~ 0 (FX forwards are nearly linear in spot).
            A 10MM EUR 1Y forward might show gamma ~ 0 USD / (spot)^2.

        Parameters
        ----------
        notional : float
            Notional in base currency.
        maturity : float
            Years to maturity.

        Returns
        -------
        float
            d2V/dS2 in domestic currency per (spot unit)^2.
        """
        bump = self.spot_rate * 0.005  # 0.5% of spot, e.g. ~54 pips for EUR/USD at 1.0850
        f_agreed = self.forward_price(maturity)  # use current fair forward as the agreed rate

        # V(S + h)
        orig_spot = self.spot_rate
        self.spot_rate = orig_spot + bump
        v_up = self.fx_swap_npv(notional, maturity, f_agreed)

        # V(S - h)
        self.spot_rate = orig_spot - bump
        v_down = self.fx_swap_npv(notional, maturity, f_agreed)

        # V(S)
        self.spot_rate = orig_spot
        v_mid = self.fx_swap_npv(notional, maturity, f_agreed)

        gamma = (v_up - 2.0 * v_mid + v_down) / (bump ** 2)
        return gamma

    def theta(self, notional: float, maturity: float) -> float:
        """Daily time decay (theta) in domestic currency.

        Computed as V(T - 1/365) - V(T), i.e. the P&L from one day passing.

        Example:
            A 10MM EUR 6M forward might have theta ~ -15 USD/day
            (the forward premium/discount decays slightly each day).

        Parameters
        ----------
        notional : float
            Notional in base currency.
        maturity : float
            Years to maturity.

        Returns
        -------
        float
            Daily P&L from time passage (domestic currency).
        """
        dt = 1.0 / 365.0
        if maturity <= dt:
            # Very short-dated: theta is just the remaining NPV
            f_agreed = self.forward_price(maturity)
            return -self.fx_swap_npv(notional, maturity, f_agreed)

        f_agreed = self.forward_price(maturity)
        v_now = self.fx_swap_npv(notional, maturity, f_agreed)
        v_later = self.fx_swap_npv(notional, maturity - dt, f_agreed)
        return v_later - v_now

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discount_domestic(self, T: float) -> float:
        """Get domestic discount factor, handling both object and callable curves."""
        return self._get_discount(self.domestic_curve, T)

    def _discount_foreign(self, T: float) -> float:
        """Get foreign discount factor."""
        return self._get_discount(self.foreign_curve, T)

    @staticmethod
    def _get_discount(curve, T: float) -> float:
        """Extract discount factor from curve object, callable, or flat rate.

        Supports:
            - curve.discount(T) method (e.g., QuantLib-style)
            - callable: curve(T)
            - float: interpreted as continuously compounded rate -> exp(-r*T)
        """
        if hasattr(curve, "discount"):
            return curve.discount(T)
        elif callable(curve):
            return curve(T)
        elif isinstance(curve, (int, float)):
            return np.exp(-curve * T)
        else:
            raise TypeError(
                f"Curve must have .discount(T), be callable, or be a flat rate (float). "
                f"Got {type(curve)}."
            )
