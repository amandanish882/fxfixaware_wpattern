"""
Market-quoted forward curve from CME FX futures
==============================================

Builds a piecewise-linear `f_market(T)` from cached CME FX futures settle
prices (Databento back-month curve) for one currency pair, with a CIP-based
fallback for tenors beyond the cached IMM grid.

This is the exchange-cleared mid quote used to mark-to-market an
FX forward / swap, replacing the textbook CIP-implied forward in places where
real market prices exist. CME settlement prices are vendor-computed mids of
the actual order book, so they already include the cross-currency basis.

Pair convention note
--------------------
This project labels the JPY pair as "JPY/USD" with spot stored as USD per JPY
(~0.006268). The CME 6J contract is also quoted USD per JPY, so no inversion
is required. EUR/USD/GBP/USD/AUD/USD all map directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class MarketForwardCurve:
    """Piecewise-linear forward curve built from CME FX futures settles.

    Parameters
    ----------
    futures_grid : pd.DataFrame
        Output of `shared.databento_curve_loader.fetch_back_month_fx_curve`,
        with columns ``["contract", "expiry_years", "price"]`` sorted by
        ascending expiry.
    spot : float
        Current FX spot in the project's pair convention.
    cip_fallback : callable, optional
        Function ``T -> float`` returning the CIP-implied forward, used for
        tenors outside the cached futures grid. Typically
        `FXPricer.forward_price`.
    """

    def __init__(
        self,
        futures_grid: pd.DataFrame,
        spot: float,
        cip_fallback: Optional[callable] = None,
    ):
        self.spot = float(spot)
        self.cip_fallback = cip_fallback

        if futures_grid is None or futures_grid.empty:
            self._T = np.array([])
            self._F = np.array([])
        else:
            g = futures_grid.sort_values("expiry_years").reset_index(drop=True)
            self._T = g["expiry_years"].to_numpy(dtype=float)
            self._F = g["price"].to_numpy(dtype=float)

    @property
    def has_cme(self) -> bool:
        return self._T.size > 0

    @property
    def t_max(self) -> float:
        return float(self._T[-1]) if self.has_cme else 0.0

    def f_market(self, T: float) -> Tuple[float, str]:
        """Return ``(forward, source)`` where source is one of:
        ``"CME"`` (interpolated between bracketing IMMs),
        ``"CME-stub"`` (front stub: spot to first IMM),
        ``"CIP"`` (CIP fallback for T beyond the cached grid).
        """
        T = float(T)

        if not self.has_cme:
            if self.cip_fallback is None:
                raise ValueError("No CME grid and no CIP fallback supplied")
            return float(self.cip_fallback(T)), "CIP"

        # Front stub: spot -> first IMM
        if T < self._T[0]:
            T_hi, F_hi = self._T[0], self._F[0]
            w = T / T_hi
            return float(self.spot + w * (F_hi - self.spot)), "CME-stub"

        # Inside grid: interpolate between bracketing IMMs
        for i in range(len(self._T) - 1):
            T_lo, T_hi = self._T[i], self._T[i + 1]
            if T_lo <= T <= T_hi:
                F_lo, F_hi = self._F[i], self._F[i + 1]
                w = (T - T_lo) / (T_hi - T_lo)
                return float(F_lo + w * (F_hi - F_lo)), "CME"

        # Beyond grid: CIP fallback
        if self.cip_fallback is None:
            # Last-resort: flat extrapolation off the final IMM.
            return float(self._F[-1]), "CME-extrap"
        return float(self.cip_fallback(T)), "CIP"
