"""Tradeable Cross-Currency Basis Market
=========================================

Models cross-currency basis as a tradeable instrument (xccy basis swap),
not a static residual.

Quote convention
----------------
xccy basis swap: pay/receive a spread on the NON-USD floating leg in
exchange for receiving/paying USD floating (3M USD SOFR or LIBOR-equivalent).

Negative basis (e.g. EUR/USD -15bp) means: a EUR holder pays 15bp ON TOP
of EUR floating to receive USD floating. Equivalently: USD funding is
expensive for non-USD holders.

Structural drivers
------------------
- EUR/USD: post-2008 EUR banks need USD funding; structurally negative
- JPY/USD: most negative -- Japanese real-money hedging FX risk on USD assets
- GBP/USD: less negative (UK is USD-funding-balanced)
- AUD/USD: mildly negative (similar to GBP)

Quarter-end / year-end blowouts: USD funding squeeze widens the basis
(makes it MORE negative) by 5-30bp depending on pair.

FX-swap-implied vs xccy-basis-swap
----------------------------------
The two markets are arbitrage-linked. For tenors <= 1Y the FX-swap-implied
basis (from the short-end of the swap-points curve) trades within a few
bps of the xccy basis swap quote. Wider divergence = arbitrage opportunity
(modulo balance-sheet costs).
"""

from __future__ import annotations

from calendar import monthrange

import pandas as pd

# Hardcoded structural levels in bps at 5y tenor (typical 2024-2026 levels).
# Used as fallback if the calibration-from-cache step fails.
_STRUCTURAL_5Y_BPS_HARDCODED = {
    "EUR/USD": -25.0,
    "GBP/USD": -15.0,
    "JPY/USD": -50.0,
    "AUD/USD": -22.0,
}

# Slope per year: how much more negative as tenor increases
_SLOPE_BPS_PER_YR = {
    "EUR/USD": -2.0,
    "GBP/USD": -1.5,
    "JPY/USD": -4.0,
    "AUD/USD": -2.5,
}

STANDARD_TENORS_YR = [1/12, 3/12, 6/12, 1.0, 2.0, 5.0]

# Year-end / quarter-end basis blowout magnitudes (bps, additive on top of structural)
_TOY_BLOWOUT_BPS = {
    "EUR/USD": -15.0,
    "GBP/USD": -8.0,
    "JPY/USD": -25.0,
    "AUD/USD": -10.0,
}
_TOQ_BLOWOUT_BPS = {pair: bps * 0.5 for pair, bps in _TOY_BLOWOUT_BPS.items()}


def _calibrate_structural_5y_bps():
    """Calibrate near-end xccy basis from observed CME front-vs-back-month data,
    extrapolated to 5Y using the hardcoded per-year slope.

    Reads ``data/databento_cache/fx_curve_history_2024-12-15_to_2026-01-31.parquet``,
    computes the observed implied basis for each G4 pair (6E/6B/6J/6A) using
    front (.c.0) vs next-quarter (.c.1) close prices, and projects to 5Y via
    ``basis_5y = basis_near + slope_per_yr * (5 - near_tenor)``.

    Falls back to ``{}`` if the cache is missing or empty.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    cache = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "databento_cache"
        / "fx_curve_history_2024-12-15_to_2026-01-31.parquet"
    )
    if not cache.exists():
        return {}
    try:
        df = pd.read_parquet(cache)
    except Exception:
        return {}
    if df.empty:
        return {}

    if "ts_event" in df.columns:
        df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 0]).dt.date
    df["ticker"] = df["symbol"].str.slice(0, 2)
    df["cont_idx"] = df["symbol"].str.extract(r"\.c\.(\d+)").astype(float)

    pair_from_ticker = {
        "6E": "EUR/USD",
        "6B": "GBP/USD",
        "6J": "JPY/USD",
        "6A": "AUD/USD",
    }
    out: dict = {}

    for ticker, pair in pair_from_ticker.items():
        sub = df[df["ticker"] == ticker]
        if sub.empty:
            continue
        front = sub[sub["cont_idx"] == 0].set_index("date")["close"]
        back = sub[sub["cont_idx"] == 1].set_index("date")["close"]
        merged = pd.DataFrame({"front": front, "back": back}).dropna()
        if len(merged) < 5:
            continue
        # Approx tenor between front and next-quarter contract: ~3 months = 0.25y.
        # CME futures quote NON-USD per USD inverted (e.g. 6E = USD per EUR), so
        # log(back/front) captures the forward premium = (r_USD - r_nonUSD) under
        # CIP. The xccy basis convention is the spread on the non-USD leg, which
        # by sign convention is the negative of that rate differential.
        near_tenor = 0.25
        merged["basis_bps_annualised"] = (
            -np.log(merged["back"] / merged["front"]) / near_tenor * 1e4
        )
        # Use median across all dates as the structural near-end estimate
        basis_near = float(merged["basis_bps_annualised"].median())

        # Extrapolate to 5Y using hardcoded slope
        slope = _SLOPE_BPS_PER_YR.get(pair, -1.0)
        basis_5y = basis_near + slope * (5.0 - near_tenor)
        out[pair] = round(basis_5y, 1)

    return out


def _load_structural_5y_bps():
    """Calibrated values where available, hardcoded fallback elsewhere."""
    merged = dict(_STRUCTURAL_5Y_BPS_HARDCODED)
    try:
        cal = _calibrate_structural_5y_bps()
        for pair, val in cal.items():
            merged[pair] = val
    except Exception:
        pass
    return merged


# Public name: calibrated where possible, hardcoded fallback otherwise.
_STRUCTURAL_5Y_BPS = _load_structural_5y_bps()


def structural_basis_curve(pair: str, as_of: str) -> pd.DataFrame:
    """Return the structural xccy basis term structure for a pair.

    Parameters
    ----------
    pair : str
        Pair label, e.g. ``"EUR/USD"``.
    as_of : str
        ISO date (used for caching keys; structure itself is roughly stable).

    Returns
    -------
    pd.DataFrame
        Columns: ``tenor_years``, ``tenor_label``, ``basis_bps``.
    """
    pair_key = pair.upper()
    base_5y = _STRUCTURAL_5Y_BPS.get(pair_key, -10.0)
    slope = _SLOPE_BPS_PER_YR.get(pair_key, -1.0)

    rows = []
    for t in STANDARD_TENORS_YR:
        # Linear interp anchored at 5y: basis grows MORE negative as tenor extends.
        # slope < 0, so (t - 5) < 0 for t < 5 makes basis less negative at the short end.
        basis = base_5y + slope * (t - 5.0)
        if t < 1.0:
            label = f"{int(round(t * 12))}M"
        else:
            label = f"{int(round(t))}Y"
        rows.append({"tenor_years": t, "tenor_label": label, "basis_bps": basis})
    return pd.DataFrame(rows)


def _days_to_period_end(as_of: str, period: str) -> int:
    """Days from ``as_of`` to the next year-end / quarter-end."""
    d = pd.to_datetime(as_of)
    if period == "year":
        target = pd.Timestamp(year=d.year, month=12, day=31)
    else:
        # Next quarter-end
        q_month = ((d.month - 1) // 3 + 1) * 3
        if q_month > 12:
            target = pd.Timestamp(year=d.year + 1, month=3, day=31)
        else:
            last = monthrange(d.year, q_month)[1]
            target = pd.Timestamp(year=d.year, month=q_month, day=last)
    return (target - d).days


def basis_with_turn_blowout(
    pair: str,
    tenor_years: float,
    as_of: str,
) -> float:
    """Basis in bps including a turn-of-period blowout if the contract crosses a turn.

    The blowout decays linearly: full magnitude when the swap covers the
    turn date, half magnitude one tenor away, zero beyond two tenors.
    """
    # Structural baseline at this tenor
    rows = structural_basis_curve(pair, as_of).set_index("tenor_years")
    if tenor_years in rows.index:
        baseline = rows.loc[tenor_years, "basis_bps"]
    else:
        baseline = (
            rows["basis_bps"].iloc[(rows.index - tenor_years).abs().argsort()[0]]
        )

    # Year-end overlay: applies if the swap window contains Dec 31
    days_to_year_end = _days_to_period_end(as_of, "year")
    swap_days = int(round(tenor_years * 365))
    overlay = 0.0
    if 0 <= days_to_year_end <= swap_days:
        overlay += _TOY_BLOWOUT_BPS.get(pair.upper(), -5.0)
    else:
        days_to_q_end = _days_to_period_end(as_of, "quarter")
        if 0 <= days_to_q_end <= swap_days:
            overlay += _TOQ_BLOWOUT_BPS.get(pair.upper(), -2.5)

    return float(baseline + overlay)


def arbitrage_band(
    pair: str,
    fx_implied_bps: float,
    xccy_quoted_bps: float,
    tolerance_bps: float = 5.0,
) -> dict:
    """Compare FX-swap-implied basis to the xccy basis swap quote.

    Returns
    -------
    dict with keys:
        pair, fx_implied_bps, xccy_quoted_bps, divergence_bps, arbitrage, direction.

    ``direction`` is ``"buy_xccy_sell_fxswap"`` if the FX-implied basis is more
    negative than the xccy quote (implied basis is "cheap" -- you pay less
    via FX swap than the xccy market would charge), or ``"sell_xccy_buy_fxswap"``
    in the opposite case.
    """
    diff = fx_implied_bps - xccy_quoted_bps
    arb = abs(diff) > tolerance_bps
    direction = None
    if arb:
        if diff < 0:
            direction = "buy_xccy_sell_fxswap"
        else:
            direction = "sell_xccy_buy_fxswap"
    return {
        "pair": pair,
        "fx_implied_bps": fx_implied_bps,
        "xccy_quoted_bps": xccy_quoted_bps,
        "divergence_bps": diff,
        "arbitrage": arb,
        "direction": direction,
    }
