"""
Generic foreign futures-strip bootstrap for any RFR currency.
=============================================================

Mirrors :func:`module_a_curves.sofr_futures_bootstrap.bootstrap_usd_ois_from_strip`
but with a configurable ``ref_period_yr`` so the same code path handles 1M
(SR1 / SOA / FEMP / IB) and 3M (SR3 / SO3 / FST3 / TY) RFR-futures strips
alike.  Accepts an arbitrary mix of contracts in one DataFrame; rows are
sorted by ``expiry_years`` and chained as simple-compound forwards over
``[expiry_years, expiry_years + ref_period_yr]``.

Schema expected
---------------
``pd.DataFrame`` with at least
    ``contract``, ``expiry_years``, ``implied_rate``
where ``expiry_years`` is the time (in years) from the valuation date to
the contract's reference-period start.

Front-stub convention
---------------------
* ``t = 0`` carries ``df = 1.0``.
* ``t = 1/365`` is anchored at ``overnight_rate`` (continuous compounding,
  consistent with daily-compounded RFR fixings over short stubs).
* If the first contract's IMM is later than ``t = 1/365``, the gap is
  discounted at ``overnight_rate`` (continuous).  This is the same pattern
  used for the USD SR3 bootstrap.

Empty-strip handling
--------------------
Returns a degenerate one-node ``DiscountCurve`` with just the t=0 anchor
when the strip is empty.  Callers that need to detect this can check
``len(curve.times) <= 2``.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from module_a_curves.curve_bootstrapper import DiscountCurve


def bootstrap_foreign_ois_from_strip(
    strip: pd.DataFrame,
    valuation_date: str,
    overnight_rate: float | None = None,
    ref_period_yr: float = 0.25,
) -> "DiscountCurve":
    """Bootstrap a foreign OIS curve from a single futures strip.

    Parameters
    ----------
    strip : pd.DataFrame
        Columns ``contract``, ``expiry_years``, ``implied_rate``.  Empty
        DataFrame produces a stub one-node curve (only the t=0 anchor).
    valuation_date : str
        ISO date string.
    overnight_rate : float, optional
        Anchor rate at ``t = 1/365``.  Falls back to the first contract's
        implied rate if not supplied.  Used both for the t=1/365 node and
        for any front-stub between ``1/365`` and the first contract's IMM.
    ref_period_yr : float, default 0.25
        Reference-period length: ``0.25`` for 3M strips (SR3, SO3, FST3,
        TY) and ``1/12`` for 1M strips (SR1, SOA, FEMP, IB).

    Returns
    -------
    DiscountCurve
    """
    val_date = dt.date.fromisoformat(valuation_date)

    if strip is None or strip.empty:
        # Degenerate: just the t=0 anchor.  Downstream code that needs a
        # usable curve must check for this.
        return DiscountCurve(
            times=[0.0, 1.0 / 365.0],
            dfs=[1.0, 1.0],
            valuation_date=val_date,
            interpolation_method="log_linear",
        )

    sorted_strip = strip.sort_values("expiry_years", ascending=True).reset_index(drop=True)

    if overnight_rate is None:
        on_rate = float(sorted_strip.iloc[0]["implied_rate"])
    else:
        on_rate = float(overnight_rate)

    times: list[float] = [0.0]
    dfs: list[float] = [1.0]

    t_on = 1.0 / 365.0
    df_on = math.exp(-on_rate * t_on)
    times.append(t_on)
    dfs.append(df_on)

    prev_t = t_on
    prev_df = df_on
    prev_rate = on_rate

    for _, row in sorted_strip.iterrows():
        ref_start = float(row["expiry_years"])
        implied_rate = float(row["implied_rate"])
        ref_end = ref_start + ref_period_yr

        if ref_end <= prev_t:
            continue

        if ref_start > prev_t:
            stub_delta = ref_start - prev_t
            df_stub = prev_df * math.exp(-prev_rate * stub_delta)
            times.append(ref_start)
            dfs.append(df_stub)
            prev_t = ref_start
            prev_df = df_stub
        elif ref_start < prev_t:
            tail_delta = ref_end - prev_t
            df_end = prev_df / (1.0 + implied_rate * tail_delta)
            times.append(ref_end)
            dfs.append(df_end)
            prev_t = ref_end
            prev_df = df_end
            prev_rate = implied_rate
            continue

        df_end = prev_df / (1.0 + implied_rate * ref_period_yr)
        times.append(ref_end)
        dfs.append(df_end)
        prev_t = ref_end
        prev_df = df_end
        prev_rate = implied_rate

    return DiscountCurve(
        times=times,
        dfs=dfs,
        valuation_date=val_date,
        interpolation_method="log_linear",
    )
