"""DTCC PPD xccy basis loader (validation supplement).

This is a thin convenience wrapper over ``shared.dtcc_sdr_loader`` that
returns one tidy DataFrame for the four G4 pairs at the standard
{3M, 6M, 1Y, 2Y} tenors required by the CME-vs-DTCC validation artifact.

Source
------
DTCC PPD "Cumulative Slice Reporting" daily ZIPs published under the
CFTC Part 43 real-time public dissemination regime:

    https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/
        CFTC_CUMULATIVE_RATES_{YYYY}_{MM}_{DD}.zip

Public dashboard: https://pddata.dtcc.com/ppd/cftcdashboard
ZIP files cached in ``data/dtcc_sdr_cache/`` by the underlying loader.

Data downloaded: 2026-04-21 .. 2026-05-05 (covers val_date 2026-04-28 +-5 BD).

Caveats (per CFTC Part 43)
--------------------------
* Block-cap truncation: notionals above the Super-Major cap (~$250M for
  IRS/xccy basis) are reported as "$250M+" with the true size hidden.
* US-person scope: only trades with at least one US-person counterparty
  are reported here. Pure inter-dealer non-US-vs-non-US trades are
  invisible.
* Trade-print, not mid: every row is an executed price including
  bid/offer, sales credits, and any block-trade discount; this is not a
  cleaned mid. We use a notional-weighted median to mitigate.
* Sparsity past 5Y is severe; standard tenors (3M/6M/1Y/2Y) are well-
  populated for EUR/USD and JPY/USD but thin for AUD/USD.
"""

from __future__ import annotations

import pandas as pd

from shared.dtcc_sdr_loader import fetch_xccy_basis_curve

# Tenors we report for the CME vs DTCC validation artifact.
_VALIDATION_TENORS = {"3M", "6M", "1Y", "2Y"}

G4_PAIRS = ("EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD")


def fetch_g4_basis_panel(
    valuation_date: str,
    window_bd: int = 5,
    rfr_only: bool = True,
) -> pd.DataFrame:
    """Fetch DTCC PPD basis medians for all G4 pairs at standard tenors.

    Parameters
    ----------
    valuation_date : str
        ISO date, e.g. ``"2026-04-28"``. Centre of the rolling window.
    window_bd : int, default 5
        Half-width of the BD window passed to the underlying loader.
    rfr_only : bool, default True
        Restrict to RFR-vs-SOFR underliers (e.g. ESTR/SOFR). When False,
        also accept IBOR-style underliers so AUD-BBSW vs USD-SOFR rows
        survive (helpful when AUD coverage is otherwise zero).

    Returns
    -------
    pd.DataFrame
        Columns: ``pair, tenor_label, tenor_years, basis_bps_dtcc,
        n_trades, weighted_notional_usd``. One row per (pair, tenor) in
        ``{3M, 6M, 1Y, 2Y}``. NaN ``basis_bps_dtcc`` if no trades.
    """
    pieces: list[pd.DataFrame] = []
    for pair in G4_PAIRS:
        df = fetch_xccy_basis_curve(
            valuation_date,
            pair,
            window_bd=window_bd,
            rfr_only=rfr_only,
        )
        df = df[df["tenor_label"].isin(_VALIDATION_TENORS)].copy()
        df = df.rename(columns={"basis_bps": "basis_bps_dtcc"})
        pieces.append(
            df[
                [
                    "pair",
                    "tenor_label",
                    "tenor_years",
                    "basis_bps_dtcc",
                    "n_trades",
                    "weighted_notional_usd",
                ]
            ]
        )
    out = pd.concat(pieces, ignore_index=True)
    return out
