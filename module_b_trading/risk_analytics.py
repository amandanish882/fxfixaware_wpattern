"""
FX Risk Analytics Engine
========================

Provides bump-and-revalue sensitivities (spot delta, rate delta, cross gamma),
portfolio-level aggregation, parametric VaR, and CME futures hedge mapping.

Example hedge mapping:
    A 10MM EUR/USD swap with delta = +96,560 USD maps to:
    6E futures (125,000 EUR/contract) -> 96,560 / (125,000 * 1.0850) ~ 0.71 -> 1 contract long
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from .fx_pricer import FXPricer, FXSwapSpec


# CME FX Futures contract specifications
CME_FX_FUTURES = {
    "EUR/USD": {"ticker": "6E", "contract_size": 125_000, "currency": "EUR"},
    "GBP/USD": {"ticker": "6B", "contract_size": 62_500, "currency": "GBP"},
    "JPY/USD": {"ticker": "6J", "contract_size": 12_500_000, "currency": "JPY"},
    "AUD/USD": {"ticker": "6A", "contract_size": 100_000, "currency": "AUD"},
}


class FXRiskAnalytics:
    """FX risk engine with bump-and-revalue Greeks and portfolio aggregation.

    Parameters
    ----------
    bootstrapper : object or None
        Curve bootstrapper that can rebuild curves after rate bumps.
        Must support ``bootstrap(instruments, bump_bps=0)`` returning a curve.
        If None, a simple flat-rate fallback is used.
    instruments : dict or None
        Instrument definitions keyed by currency, e.g.
        {"USD": [...], "EUR": [...]}.
    valuation_date : str or datetime
        The as-of date for valuations.
    fx_spots : dict
        Spot rates keyed by pair, e.g. {"EUR/USD": 1.0850, "GBP/USD": 1.2650}.
    """

    def __init__(self, bootstrapper, instruments, valuation_date, fx_spots: Dict[str, float]):
        self.bootstrapper = bootstrapper
        self.instruments = instruments
        self.valuation_date = valuation_date
        self.fx_spots = fx_spots

    # ------------------------------------------------------------------
    # Spot delta via bump-and-revalue
    # ------------------------------------------------------------------

    def delta_spot(self, swap: FXSwapSpec) -> float:
        """Spot delta via +/-10 pip bump-and-revalue.

        For EUR/USD at 1.0850, bumps to 1.0860 and 1.0840 (10 pips = 0.0010).
        Delta = (V_up - V_down) / (2 * bump).

        Example:
            10MM EUR/USD 1Y swap, agreed fwd = 1.1000
            V(1.0860) - V(1.0840) ~ 9,656 USD difference
            delta = 9,656 / 0.0020 ~ 4,828,000 USD per 1 full spot point

        Parameters
        ----------
        swap : FXSwapSpec
            The FX swap to analyse.

        Returns
        -------
        float
            Spot delta in domestic currency per 1 unit of spot move.
        """
        pair = swap.pair
        spot = self.fx_spots.get(pair, 1.0)

        # 10 pips: 0.0010 for most pairs, 0.10 for JPY-quoted pairs
        if "JPY" in pair:
            bump = 0.10  # 10 pips for JPY pairs (e.g. USD/JPY at 149.50 -> 149.60)
        else:
            bump = 0.0010  # 10 pips for EUR/USD, GBP/USD, AUD/USD

        pricer_up = self._build_pricer(pair, spot + bump)
        pricer_down = self._build_pricer(pair, spot - bump)

        sign = swap.sign()
        v_up = sign * pricer_up.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
        v_down = sign * pricer_down.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)

        return (v_up - v_down) / (2.0 * bump)

    # ------------------------------------------------------------------
    # Rate delta via curve rebuild
    # ------------------------------------------------------------------

    def delta_rates(self, swap: FXSwapSpec, curve_type: str = "domestic") -> float:
        """Interest rate delta via +/-1bp parallel bump, curve rebuild, reprice.

        Bumps the domestic (or foreign) curve by +/-1bp, rebuilds, then reprices.

        Example:
            10MM EUR/USD 1Y swap:
            If USD rates bump +1bp (5.25% -> 5.26%), the forward moves by ~0.0001
            rate_delta ~ 10e6 * 0.0001 * D(1) ~ 950 USD per bp

        Parameters
        ----------
        swap : FXSwapSpec
            The FX swap to analyse.
        curve_type : str
            'domestic' or 'foreign' - which curve to bump.

        Returns
        -------
        float
            Rate delta in domestic currency per 1bp parallel shift.
        """
        pair = swap.pair
        spot = self.fx_spots.get(pair, 1.0)
        bump_bps = 1.0  # 1 basis point = 0.0001 in rate terms

        if self.bootstrapper is not None and self.instruments is not None:
            base_ccy, quote_ccy = self._parse_pair(pair)
            target_ccy = quote_ccy if curve_type == "domestic" else base_ccy

            if target_ccy in self.instruments:
                try:
                    curve_up = self.bootstrapper.bootstrap(
                        self.instruments[target_ccy], bump_bps=bump_bps
                    )
                    curve_down = self.bootstrapper.bootstrap(
                        self.instruments[target_ccy], bump_bps=-bump_bps
                    )

                    base_dom = self._get_curve(quote_ccy)
                    base_for = self._get_curve(base_ccy)

                    if curve_type == "domestic":
                        p_up = FXPricer(curve_up, base_for, spot, pair)
                        p_down = FXPricer(curve_down, base_for, spot, pair)
                    else:
                        p_up = FXPricer(base_dom, curve_up, spot, pair)
                        p_down = FXPricer(base_dom, curve_down, spot, pair)

                    sign = swap.sign()
                    v_up = sign * p_up.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
                    v_down = sign * p_down.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
                    return (v_up - v_down) / 2.0
                except Exception:
                    pass

        # Fallback: analytical approximation
        # rate_delta ~ notional * T * D(T) * spot * 0.0001
        T = swap.maturity_years
        df = np.exp(-0.05 * T)  # assume ~5% rate
        return swap.sign() * swap.notional * T * df * spot * 1e-4

    # ------------------------------------------------------------------
    # Cross gamma (d2V/dS2)
    # ------------------------------------------------------------------

    def cross_gamma(self, swap: FXSwapSpec) -> float:
        """Numerical d2V/dS2 via three-point finite difference on spot.

        Example:
            For a vanilla FX forward, gamma is near zero because the payoff
            is linear in spot. A 10MM EUR/USD 1Y forward: gamma ~ 0.

        Parameters
        ----------
        swap : FXSwapSpec
            The FX swap to analyse.

        Returns
        -------
        float
            d2V/dS2 in domestic currency per (spot)^2.
        """
        pair = swap.pair
        spot = self.fx_spots.get(pair, 1.0)

        if "JPY" in pair:
            bump = 0.10
        else:
            bump = 0.0010

        p_up = self._build_pricer(pair, spot + bump)
        p_mid = self._build_pricer(pair, spot)
        p_down = self._build_pricer(pair, spot - bump)

        sign = swap.sign()
        v_up = sign * p_up.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
        v_mid = sign * p_mid.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
        v_down = sign * p_down.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)

        return (v_up - 2.0 * v_mid + v_down) / (bump ** 2)

    # ------------------------------------------------------------------
    # Portfolio aggregation
    # ------------------------------------------------------------------

    def portfolio_risk(self, portfolio: List[FXSwapSpec]) -> pd.DataFrame:
        """Aggregate spot and rate deltas across a portfolio of FX swaps.

        Example output:
            pair      | spot_delta  | dom_rate_delta | for_rate_delta | notional
            EUR/USD   | 482,800     | 950            | -920           | 10,000,000
            GBP/USD   | 312,500     | 620            | -600           | 5,000,000

        Parameters
        ----------
        portfolio : list of FXSwapSpec
            Collection of FX swap positions.

        Returns
        -------
        pd.DataFrame
            Risk summary per currency pair with aggregated deltas.
        """
        rows = []
        for swap in portfolio:
            rows.append({
                "pair": swap.pair,
                "direction": swap.direction,
                "notional": swap.notional,
                "maturity_years": swap.maturity_years,
                "spot_delta": self.delta_spot(swap),
                "dom_rate_delta": self.delta_rates(swap, "domestic"),
                "for_rate_delta": self.delta_rates(swap, "foreign"),
            })

        df = pd.DataFrame(rows)

        # Aggregate by pair
        if len(df) > 0:
            summary = df.groupby("pair").agg(
                total_notional=("notional", "sum"),
                net_spot_delta=("spot_delta", "sum"),
                net_dom_rate_delta=("dom_rate_delta", "sum"),
                net_for_rate_delta=("for_rate_delta", "sum"),
                n_trades=("pair", "count"),
            ).reset_index()
            return summary

        return pd.DataFrame(columns=[
            "pair", "total_notional", "net_spot_delta",
            "net_dom_rate_delta", "net_for_rate_delta", "n_trades"
        ])

    # ------------------------------------------------------------------
    # Correlation and VaR
    # ------------------------------------------------------------------

    def correlation_matrix(self, returns_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Estimate cross-pair correlation from trailing 252 daily returns.

        If no returns history is provided, uses realistic default correlations
        calibrated to major FX pairs.

        Example default matrix (approximate):
                    EUR/USD  GBP/USD  JPY/USD  AUD/USD
            EUR/USD   1.00     0.70    -0.30     0.55
            GBP/USD   0.70     1.00    -0.25     0.50
            JPY/USD  -0.30    -0.25     1.00    -0.35
            AUD/USD   0.55     0.50    -0.35     1.00

        Parameters
        ----------
        returns_history : pd.DataFrame, optional
            Columns are pair names, rows are daily returns.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        if returns_history is not None and len(returns_history) >= 20:
            return returns_history.corr()

        # Realistic default correlations for G4 FX pairs vs USD
        pairs = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]
        corr = np.array([
            [1.00, 0.70, -0.30, 0.55],
            [0.70, 1.00, -0.25, 0.50],
            [-0.30, -0.25, 1.00, -0.35],
            [0.55, 0.50, -0.35, 1.00],
        ])
        return pd.DataFrame(corr, index=pairs, columns=pairs)

    def var_portfolio(self, portfolio: List[FXSwapSpec],
                      confidence: float = 0.99,
                      returns_history: Optional[pd.DataFrame] = None) -> float:
        """Parametric (delta-normal) Value-at-Risk for an FX portfolio.

        VaR = z * sqrt(delta' * Sigma * delta)

        where delta is the vector of spot deltas and Sigma is the covariance
        matrix of daily spot returns.

        Example:
            Portfolio with EUR/USD delta = +482,800 and GBP/USD delta = +312,500
            Daily vol: EUR ~0.55%, GBP ~0.65%, correlation 0.70
            VaR(99%) ~ 2.326 * sqrt(combined variance) ~ ~25,000 USD

        Parameters
        ----------
        portfolio : list of FXSwapSpec
            Portfolio positions.
        confidence : float
            VaR confidence level, e.g. 0.99 for 99% VaR.

        Returns
        -------
        float
            1-day parametric VaR in domestic currency (positive number = loss).
        """
        from scipy.stats import norm

        risk_df = self.portfolio_risk(portfolio)
        if len(risk_df) == 0:
            return 0.0

        corr_mat = self.correlation_matrix(returns_history)
        pairs_in_portfolio = risk_df["pair"].tolist()

        # Daily volatilities (annualized ~8-12% for majors -> daily ~0.5-0.75%)
        default_daily_vols = {
            "EUR/USD": 0.0055,   # ~8.7% annualized
            "GBP/USD": 0.0065,   # ~10.3% annualized
            "JPY/USD": 0.0060,   # ~9.5% annualized
            "AUD/USD": 0.0070,   # ~11.1% annualized
        }

        n = len(pairs_in_portfolio)
        delta_vec = np.zeros(n)
        vol_vec = np.zeros(n)
        corr_sub = np.eye(n)

        for i, pair in enumerate(pairs_in_portfolio):
            delta_vec[i] = risk_df.loc[risk_df["pair"] == pair, "net_spot_delta"].values[0]
            spot = self.fx_spots.get(pair, 1.0)
            # Convert delta from "per spot unit" to "per % move" using daily vol
            daily_vol = default_daily_vols.get(pair, 0.006)
            vol_vec[i] = daily_vol * spot  # daily spot move in absolute terms

            for j, pair_j in enumerate(pairs_in_portfolio):
                if pair in corr_mat.index and pair_j in corr_mat.columns:
                    corr_sub[i, j] = corr_mat.loc[pair, pair_j]

        # Covariance matrix: Sigma = diag(vol) * Corr * diag(vol)
        cov_mat = np.outer(vol_vec, vol_vec) * corr_sub

        # Portfolio variance: delta' * Sigma * delta
        portfolio_var = delta_vec @ cov_mat @ delta_vec

        if portfolio_var < 0:
            portfolio_var = 0.0

        z = norm.ppf(confidence)  # e.g. 2.326 for 99%
        var_1d = z * np.sqrt(portfolio_var)
        return abs(var_1d)

    # ------------------------------------------------------------------
    # CME futures hedge
    # ------------------------------------------------------------------

    def best_futures_hedge(self, swap: FXSwapSpec) -> Dict[str, Any]:
        """Map an FX swap's spot delta to the best CME FX futures hedge.

        Example:
            10MM EUR/USD swap with spot_delta = 4,828,000 USD per spot point
            6E contract = 125,000 EUR -> dollar value per contract = 125,000 * 1.0850 = 135,625
            n_contracts = round(4,828,000 / 135,625) = round(35.6) = 36 contracts
            direction = 'short' (to offset long delta)

        Parameters
        ----------
        swap : FXSwapSpec
            The FX swap position to hedge.

        Returns
        -------
        dict
            Keys: ticker, n_contracts (int), direction ('long'/'short'),
            contract_size, residual_delta (unhedged remainder).
        """
        pair = swap.pair
        if pair not in CME_FX_FUTURES:
            return {
                "ticker": None,
                "n_contracts": 0,
                "direction": "flat",
                "contract_size": 0,
                "residual_delta": 0.0,
                "error": f"No CME futures mapping for {pair}",
            }

        spec = CME_FX_FUTURES[pair]
        spot = self.fx_spots.get(pair, 1.0)
        spot_delta = self.delta_spot(swap)

        # Dollar value of one futures contract
        # For EUR/USD: 125,000 EUR * 1.0850 USD/EUR = 135,625 USD per contract
        contract_dollar_value = spec["contract_size"] * spot

        if contract_dollar_value == 0:
            return {"ticker": spec["ticker"], "n_contracts": 0, "direction": "flat",
                    "contract_size": spec["contract_size"], "residual_delta": spot_delta}

        n_contracts_raw = spot_delta / contract_dollar_value
        n_contracts = int(round(abs(n_contracts_raw)))

        # Hedge opposes the delta: if delta is positive (long), hedge is short
        direction = "short" if n_contracts_raw > 0 else "long"

        hedged_delta = n_contracts * contract_dollar_value * (1 if direction == "long" else -1)
        residual = spot_delta + hedged_delta

        return {
            "ticker": spec["ticker"],
            "n_contracts": n_contracts,
            "direction": direction,
            "contract_size": spec["contract_size"],
            "residual_delta": residual,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pricer(self, pair: str, spot: float) -> FXPricer:
        """Construct an FXPricer for the given pair and spot rate."""
        base_ccy, quote_ccy = self._parse_pair(pair)
        dom_curve = self._get_curve(quote_ccy)
        for_curve = self._get_curve(base_ccy)
        return FXPricer(dom_curve, for_curve, spot, pair)

    def _get_curve(self, ccy: str):
        """Retrieve or build a discount curve for the given currency.

        Falls back to realistic flat rates if no bootstrapper is available.
        """
        # Realistic fallback rates (mid-2024 levels)
        fallback_rates = {
            "USD": 0.0525,  # Fed funds ~5.25%
            "EUR": 0.0350,  # ECB deposit ~3.50%
            "GBP": 0.0500,  # BoE base ~5.00%
            "JPY": 0.0010,  # BoJ ~0.10%
            "AUD": 0.0425,  # RBA cash ~4.25%
        }

        if self.bootstrapper is not None and self.instruments is not None:
            if ccy in self.instruments:
                try:
                    return self.bootstrapper.bootstrap(self.instruments[ccy])
                except Exception:
                    pass

        # Return flat rate as a float; FXPricer._get_discount handles exp(-r*T)
        return fallback_rates.get(ccy, 0.04)

    @staticmethod
    def _parse_pair(pair: str):
        """Parse 'EUR/USD' into ('EUR', 'USD')."""
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair}. Expected 'XXX/YYY'.")
        return parts[0], parts[1]
