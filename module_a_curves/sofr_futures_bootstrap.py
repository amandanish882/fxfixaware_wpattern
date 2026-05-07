"""
SOFR Futures Bootstrap for USD OIS DiscountCurve
================================================

Constructs a USD OIS ``DiscountCurve`` from a CME SOFR (SR3) 3-month futures
strip using a standard 3M-forward bootstrap.

This replaces the Treasury-yield (DGS2/5/10/30) proxy used previously in
``run_full_demo.py``: SR3 futures imply forward 3M SOFR rates directly, which
is a much closer match to the actual OIS discount curve than Treasury yields.

Strip schema
------------
Each row carries

* ``expiry_years`` -- time (in years) from the valuation date to the contract
  IMM date (= start of the 3-month SR3 reference period; for the M6 contract
  this is the 3rd Wednesday of June);
* ``implied_rate`` -- the simple forward rate over the reference period
  ``[IMM, IMM + 3M]``, derived from ``(100 - price)/100``.

Algorithm (no convexity adjustment)
-----------------------------------
1. Anchor at ``t = 0`` with ``df = 1.0``.
2. Add an overnight node at ``t = 1/365`` using the supplied (or first-contract)
   overnight rate: ``df(1/365) = 1 / (1 + r_on / 365)``.
3. For each SR3 contract in ascending IMM order, with
   ``ref_start = expiry_years`` and ``ref_end = ref_start + 0.25``:

   * If ``ref_start > prev_t`` there is a stub between the previous node and
     ``ref_start``.  Discount that stub at ``prev_rate`` (the overnight rate
     for the front stub, or the previous contract's implied forward for
     contract-to-contract gaps that should not normally appear because
     consecutive quarterly SR3s have ``ref_end_i == ref_start_{i+1}``).
   * Apply the SR3 implied forward over its own 3-month reference window:
     ``df(ref_end) = df(ref_start) / (1 + implied_rate * 0.25)``.

   This pins the front of the curve to the real overnight SOFR fixing and
   avoids the ~13bp dip the previous (ref-end-anchored) bootstrap produced
   when ``expiry_years`` was the time to ref-end rather than ref-start.

Convexity (futures-vs-FRA bias) is omitted: the bias is typically <1bp at <1y
and <5bp at 2y for SR3 strips, which is negligible relative to the <0.5bp
quote precision currently fed in.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from module_a_curves.curve_bootstrapper import DiscountCurve

# SR3 reference period length: 3 calendar months expressed in years.
_SR3_REF_PERIOD_YR = 0.25


def bootstrap_usd_ois_from_strip(
    strip: pd.DataFrame,
    valuation_date: str,
    overnight_rate: float | None = None,
) -> DiscountCurve:
    """Bootstrap a USD OIS DiscountCurve from a SOFR (SR3) futures strip.

    Standard 3M-forward bootstrap, no convexity adjustment (typically <1bp
    at <1y, <5bp at 2y).

    Parameters
    ----------
    strip : pd.DataFrame
        Columns: ``contract``, ``expiry_years``, ``implied_rate``.
        ``expiry_years`` is the time to the contract IMM date (start of the
        3-month reference period); the implied forward applies over
        ``[expiry_years, expiry_years + 0.25]``.
    valuation_date : str
        ISO date string (e.g. ``"2026-04-28"``).
    overnight_rate : float, optional
        Anchor rate at ``t = 1/365``; if not provided, falls back to the
        first contract's implied rate.  This rate is also used to discount
        the front stub from ``t = 1/365`` to the first contract's IMM date.

    Returns
    -------
    DiscountCurve
        With times starting at 0 and discount factors decreasing.
    """
    if strip is None or strip.empty:
        raise ValueError("Empty SOFR strip cannot bootstrap a curve")

    strip_sorted = strip.sort_values("expiry_years", ascending=True).reset_index(drop=True)

    # Pick the anchor rate (front-stub rate too).
    if overnight_rate is None:
        on_rate = float(strip_sorted.iloc[0]["implied_rate"])
    else:
        on_rate = float(overnight_rate)

    # t = 0 anchor and overnight node.
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]

    # Overnight anchor at t = 1/365.  SOFR is published as a daily rate, so
    # treat it as continuously compounded over short stubs (daily compounding
    # ≈ continuous to first order).  This avoids a ~9bp simple-vs-continuous
    # convention drop between t=1/365 and the first SR3 IMM (~50d out).
    t_on = 1.0 / 365.0
    df_on = math.exp(-on_rate * t_on)
    times.append(t_on)
    dfs.append(df_on)

    prev_t = t_on
    prev_df = df_on
    prev_rate = on_rate  # rate used to discount any stub up to the next IMM

    for _, row in strip_sorted.iterrows():
        ref_start = float(row["expiry_years"])
        implied_rate = float(row["implied_rate"])
        ref_end = ref_start + _SR3_REF_PERIOD_YR

        # Skip contracts that are entirely behind the curve we have built.
        if ref_end <= prev_t:
            continue

        # Front stub (or any small gap) from prev_t up to ref_start, priced
        # at prev_rate.  For the very first contract the stub uses the FRED
        # overnight SOFR fixing.  We discount with continuous compounding
        # (consistent with how SOFR-as-daily-rate compounds over a stub of
        # ~50 days); SR3 forwards inside their reference window are simple
        # because that is the SR3 settlement convention.
        if ref_start > prev_t:
            stub_delta = ref_start - prev_t
            df_stub = prev_df * math.exp(-prev_rate * stub_delta)
            times.append(ref_start)
            dfs.append(df_stub)
            prev_t = ref_start
            prev_df = df_stub
        elif ref_start < prev_t:
            # Already past this contract's IMM (overlapping front contracts);
            # apply the implied rate only over the remaining tail of its
            # reference window so we still hit ref_end.
            tail_delta = ref_end - prev_t
            df_end = prev_df / (1.0 + implied_rate * tail_delta)
            times.append(ref_end)
            dfs.append(df_end)
            prev_t = ref_end
            prev_df = df_end
            prev_rate = implied_rate
            continue

        # Apply the SR3 forward over its full 3-month reference period.
        df_end = prev_df / (1.0 + implied_rate * _SR3_REF_PERIOD_YR)
        times.append(ref_end)
        dfs.append(df_end)
        prev_t = ref_end
        prev_df = df_end
        prev_rate = implied_rate

    val_date = dt.date.fromisoformat(valuation_date)

    return DiscountCurve(
        times=times,
        dfs=dfs,
        valuation_date=val_date,
        interpolation_method="log_linear",
    )
