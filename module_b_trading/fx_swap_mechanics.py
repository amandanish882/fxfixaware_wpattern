"""FX Swap Mechanics
====================

Forward points, tomorrow-next (T/N) rolls, and turn-of-period overlays for
short-dated FX swap pricing.

Conventions
-----------
A pair "BASE/QUOTE" means: 1 BASE = quote_units of QUOTE.
For USD/JPY 150.00, 1 USD = 150 JPY. base=USD, quote=JPY.
For EUR/USD 1.0832, 1 EUR = 1.0832 USD. base=EUR, quote=USD.

The standard CIP forward (no basis) is:

    F = S * (1 + rate_quote * T) / (1 + rate_base * T)

When the QUOTE currency has the LOWER rate, F < S and forward points are
NEGATIVE -- the quote currency is at a premium (worth more in the future).

Worked example (USD/JPY, 3M, USD 4.5%, JPY 0.25%):
    F = 150 * (1 + 0.0025*0.25) / (1 + 0.045*0.25) = 148.42
    Points = -1.58 = -158 pips (pip = 0.01 for JPY pairs).
    JPY at premium -- consistent with structural Japanese USD-asset hedging flow.

Pip factor convention
---------------------
For pairs where the quote currency is JPY (or where one side is JPY),
1 pip = 0.01, so pip_factor = 100.
For majors (EUR/USD, GBP/USD, AUD/USD, etc.), 1 pip = 0.0001, pip_factor = 10000.
"""

from __future__ import annotations

import pandas as pd


def pip_factor(pair: str) -> int:
    """Return the pip multiplier for a currency pair.

    1 pip = 0.01 for JPY pairs (factor 100), 0.0001 otherwise (factor 10000).
    """
    parts = pair.upper().split("/")
    if "JPY" in parts:
        return 100
    return 10000


def forward_points(
    spot: float,
    rate_base: float,
    rate_quote: float,
    tenor_years: float,
    pair: str,
) -> "tuple[float, float]":
    """Compute the CIP forward and forward points (in pips).

    Parameters
    ----------
    spot : float
        Current spot rate, e.g. 150.00 for USD/JPY.
    rate_base : float
        Base-currency funding rate (e.g. USD 3M OIS for USD/JPY).
    rate_quote : float
        Quote-currency funding rate (e.g. JPY 3M OIS for USD/JPY).
    tenor_years : float
        Tenor in years, e.g. 0.25 for 3M.
    pair : str
        Pair label, e.g. ``"USD/JPY"`` or ``"EUR/USD"``.

    Returns
    -------
    (forward, points_pips) : tuple[float, float]
        forward = CIP forward rate
        points_pips = (forward - spot) * pip_factor(pair), signed
    """
    fwd = spot * (1.0 + rate_quote * tenor_years) / (1.0 + rate_base * tenor_years)
    pf = pip_factor(pair)
    return fwd, (fwd - spot) * pf


def tn_roll_points(
    spot: float,
    rate_base: float,
    rate_quote: float,
    pair: str,
    days: int = 1,
) -> float:
    """Tomorrow-Next swap points for an N-day roll (default 1 day).

    Spot in FX settles T+2 by convention. T/N is a 1-day swap from T+1 to T+2,
    used to fund a position one day forward at the prevailing rate differential.

    Returns
    -------
    float
        T/N forward points in pips, signed.
    """
    tenor = days / 365.0
    _, pts = forward_points(spot, rate_base, rate_quote, tenor, pair)
    return pts


# Empirical turn-of-period widening (in pips per pair). Larger for JPY pairs
# because the USD-funding squeeze hits Japanese bank balance-sheet constraints.
def _load_turn_of_year_bps() -> dict:
    """Load calibrated TOY widening from observed CME data; fall back to defaults."""
    fallback = {
        "USD/JPY": -25.0,
        "JPY/USD": -25.0,
        "EUR/USD": -8.0,
        "GBP/USD": -5.0,
        "AUD/USD": -6.0,
    }
    try:
        from module_b_trading.toy_calibration import calibrate_toy_overlay

        calibrated = calibrate_toy_overlay()
        merged = dict(fallback)
        for pair, val in calibrated.items():
            merged[pair] = val
            # Mirror to the inverted-quote form for USD/JPY <-> JPY/USD.
            # Reciprocal pairs require a sign flip on the bps differential.
            if pair == "JPY/USD":
                merged["USD/JPY"] = -val
            elif pair == "USD/JPY":
                merged["JPY/USD"] = -val
        return merged
    except Exception:
        return fallback


TURN_OF_YEAR_BPS = _load_turn_of_year_bps()
TURN_OF_QUARTER_BPS = {pair: bps * 0.4 for pair, bps in TURN_OF_YEAR_BPS.items()}


def _crosses_period(start_date: str, end_date: str, period: str) -> bool:
    """Return True if [start, end] crosses a year-end (period='year') or
    quarter-end (period='quarter')."""
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    if period == "year":
        return s.year != e.year
    if period == "quarter":
        return (s.year, (s.month - 1) // 3) != (e.year, (e.month - 1) // 3)
    return False


def apply_turn_overlay(
    base_points: float,
    start_date: str,
    end_date: str,
    pair: str,
) -> float:
    """Add a turn-of-period spike to the base forward points.

    Parameters
    ----------
    base_points : float
        Forward points before overlay, in pips.
    start_date, end_date : str
        ISO dates bounding the swap.
    pair : str
        Pair label.

    Returns
    -------
    float
        Adjusted forward points (signed).
    """
    adj = base_points
    if _crosses_period(start_date, end_date, "year"):
        adj += TURN_OF_YEAR_BPS.get(pair.upper(), -5.0)
    elif _crosses_period(start_date, end_date, "quarter"):
        adj += TURN_OF_QUARTER_BPS.get(pair.upper(), -2.0)
    return adj
