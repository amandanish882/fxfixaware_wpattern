"""
Fix-Aware FX Quote Optimizer
=============================

Optimizes quoted spreads for FX RFQs by maximizing expected P&L subject to
risk and fix-timing adjustments.

Objective:
    E[PnL](s) = P(hit|s) * (s - cost) - lambda * delta^2 + fix_adjustment

where:
    s = quoted half-spread in pips
    P(hit|s) = win probability from the logistic model
    cost = estimated hedging / execution cost (~0.1-0.3 pips for majors)
    lambda = risk aversion penalty on delta squared
    fix_adjustment = widen during fix (adverse selection), tighten pre-fix (signal)

Guardrails:
    - min_spread_breach: spread below 0.1 pips (below exchange spread)
    - high_notional_warning: notional > 50MM USD
    - fix_window_widening: forced widening during fix +-5 min
    - low_probability_reject: P(hit) < 5% at any spread -> don't quote
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from .fix_alpha_signals import FixAlphaModel
from datetime import datetime


class FXQuoteOptimizer:
    """Fix-aware quote optimizer for FX market making.

    Parameters
    ----------
    win_model : FXWinProbabilityModel
        Trained win probability model.
    fix_alpha_model : FixAlphaModel
        Fix timing alpha model.
    lambda_risk : float
        Risk aversion parameter for delta penalty. Default 1e-6.
        Higher values -> wider spreads (more conservative).

    Example:
        optimizer = FXQuoteOptimizer(win_model, fix_alpha, lambda_risk=1e-6)
        result = optimizer.optimal_quote(rfq_features, current_utc_time=datetime(2024,3,15,15,45))
        # result = {
        #     'optimal_spread_pips': 0.85,
        #     'expected_pnl': 0.42,
        #     'hit_prob': 0.72,
        #     'guardrails': [],
        # }
    """

    # Execution cost estimates in pips (half-spread cost of hedging)
    HEDGE_COST_PIPS = {
        "EUR/USD": 0.10,  # Most liquid pair
        "GBP/USD": 0.15,
        "JPY/USD": 0.12,
        "AUD/USD": 0.20,
    }

    MIN_SPREAD_PIPS = 0.1
    HIGH_NOTIONAL_THRESHOLD = 50_000_000  # 50MM USD

    def __init__(self, win_model, fix_alpha_model: Optional[FixAlphaModel] = None,
                 lambda_risk: float = 1e-6):
        self.win_model = win_model
        self.fix_alpha = fix_alpha_model or FixAlphaModel()
        self.lambda_risk = lambda_risk

    # ------------------------------------------------------------------
    # Core optimizer
    # ------------------------------------------------------------------

    def optimal_quote(
        self,
        rfq_features: pd.DataFrame,
        current_utc_time: Optional[datetime] = None,
        alpha_skew_pips: float = 0.0,
    ) -> Dict[str, Any]:
        """Find the optimal spread for an RFQ via grid search.

        Grid: 100 spreads from 0.1 to 5.0 pips.
        For each spread s:
            1. Set the spread in rfq_features
            2. Predict P(hit|s)
            3. Compute E[PnL] = P(hit) * (s - cost) - lambda * delta^2 + fix_adj
            4. Pick the spread that maximizes E[PnL]

        Example at 15:45 UTC (pre-London fix):
            Grid search finds optimal at s=0.72 pips:
            P(hit|0.72) = 0.78, cost = 0.10, fix_adj = +0.05 (tighten pre-fix)
            E[PnL] = 0.78 * (0.72 - 0.10) + 0.05 = 0.533 pips

        Parameters
        ----------
        rfq_features : pd.DataFrame
            Single-row (or multi-row) DataFrame with RFQ features.
            Must have: timestamp, pair, client_segment, notional_usd, quoted_spread_pips.
        current_utc_time : datetime, optional
            Override UTC time for fix schedule. Defaults to timestamp in data.
        alpha_skew_pips : float
            Additional skew from composite alpha model (positive = tighten bid).

        Returns
        -------
        dict
            optimal_spread_pips, expected_pnl, hit_prob, guardrails, fix_schedule.
        """
        # Work with first row if multi-row
        row = rfq_features.iloc[0] if len(rfq_features) > 0 else rfq_features

        pair = row.get("pair", "EUR/USD")
        notional = row.get("notional_usd", 5_000_000)
        hedge_cost = self.HEDGE_COST_PIPS.get(pair, 0.15)

        # Get fix schedule
        if current_utc_time is None:
            ts = row.get("timestamp", datetime.now(tz=None))
            if isinstance(ts, str):
                ts = pd.Timestamp(ts).to_pydatetime()
            current_utc_time = ts

        fix_sched = self.fix_alpha.fix_schedule(current_utc_time)

        # Fix adjustment to E[PnL]
        fix_adj = self._fix_adjustment(fix_sched, alpha_skew_pips)

        # Delta penalty (notional-based)
        delta_penalty = self.lambda_risk * (notional ** 2) * 1e-14  # scale to pips

        # Grid search over 50 spread values (vectorized, no per-point sklearn)
        spread_grid = np.linspace(0.1, 5.0, 50)

        # Get baseline P(hit) at the current spread to calibrate the logistic
        base_spread = row.get("quoted_spread_pips", 1.0)
        try:
            base_prob = self.win_model.predict(rfq_features)[0]
        except Exception:
            base_prob = 1.0 / (1.0 + np.exp(1.2 * (base_spread - 1.5)))

        # Estimate spread sensitivity: spread_coef from model, or default -1.2
        spread_coef = -1.2
        if hasattr(self.win_model, 'model') and self.win_model.model is not None:
            try:
                coefs = self.win_model.model.coef_[0]
                cols = self.win_model.feature_columns
                if cols is not None and "spread_pips" in list(cols):
                    idx = list(cols).index("spread_pips")
                    spread_coef = coefs[idx]
            except Exception:
                pass

        # Analytical P(hit) across grid: shift logit by spread_coef * (s - base_spread)
        base_logit = np.log(base_prob / (1 - base_prob + 1e-15))
        logits = base_logit + spread_coef * (spread_grid - base_spread)
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Vectorized E[PnL]
        epnls = probs * (spread_grid - hedge_cost) - delta_penalty + fix_adj

        best_idx = np.argmax(epnls)
        best_spread = spread_grid[best_idx]
        best_epnl = epnls[best_idx]
        best_prob = probs[best_idx]

        # Apply guardrails
        guardrails = self._check_guardrails(best_spread, notional, best_prob, fix_sched)

        # Enforce guardrails
        if "fix_window_widening" in guardrails:
            best_spread = max(best_spread, best_spread * fix_sched["spread_multiplier"])
        if "min_spread_breach" in guardrails:
            best_spread = max(best_spread, self.MIN_SPREAD_PIPS)

        return {
            "optimal_spread_pips": round(best_spread, 3),
            "expected_pnl": round(best_epnl, 4),
            "hit_prob": round(best_prob, 4),
            "guardrails": guardrails,
            "fix_schedule": fix_sched,
            "hedge_cost_pips": hedge_cost,
            "alpha_skew_pips": round(alpha_skew_pips, 3),
        }

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(self, rfq_data: pd.DataFrame) -> Dict:
        """Backtest the optimizer across a historical RFQ dataset.

        For each RFQ:
            1. Find optimal spread
            2. Compare to actual outcome (was_hit)
            3. Compute realized P&L

        Example output:
            {
                'total_expected_pnl': 4250.5,   # pips across all RFQs
                'avg_spread_pips': 0.92,
                'avg_hit_prob': 0.68,
                'n_rfqs': 10000,
                'guardrails_count': {'fix_window_widening': 850, ...},
            }

        Parameters
        ----------
        rfq_data : pd.DataFrame
            Historical RFQ data with was_hit column.

        Returns
        -------
        dict
            Backtest summary statistics.
        """
        # Vectorized backtest: batch-predict once, then analytical grid per row
        n = len(rfq_data)
        if n == 0:
            return {"total_expected_pnl": 0, "avg_spread": 0,
                    "avg_hit_prob": 0, "n_rfqs": 0, "n_guardrails_triggered": 0}

        # Batch predict baseline probabilities for all RFQs at once
        try:
            base_probs = self.win_model.predict(rfq_data)
        except Exception:
            spreads = rfq_data["quoted_spread_pips"].values if "quoted_spread_pips" in rfq_data.columns else np.ones(n)
            base_probs = 1.0 / (1.0 + np.exp(1.2 * (spreads - 1.5)))

        base_spreads = rfq_data["quoted_spread_pips"].values if "quoted_spread_pips" in rfq_data.columns else np.ones(n)
        pairs = rfq_data["pair"].values if "pair" in rfq_data.columns else np.full(n, "EUR/USD")
        notionals = rfq_data["notional_usd"].values if "notional_usd" in rfq_data.columns else np.full(n, 5e6)
        was_hits = rfq_data["was_hit"].values if "was_hit" in rfq_data.columns else np.zeros(n, dtype=bool)

        # Get spread sensitivity
        spread_coef = -1.2
        if hasattr(self.win_model, 'model') and self.win_model.model is not None:
            try:
                coefs = self.win_model.model.coef_[0]
                cols = self.win_model.feature_columns
                if cols is not None and "spread_pips" in list(cols):
                    idx = list(cols).index("spread_pips")
                    spread_coef = coefs[idx]
            except Exception:
                pass

        spread_grid = np.linspace(0.1, 5.0, 50)
        opt_spreads = np.zeros(n)
        opt_epnls = np.zeros(n)
        opt_probs = np.zeros(n)
        guardrail_count = 0

        for i in range(n):
            hedge_cost = self.HEDGE_COST_PIPS.get(pairs[i], 0.15)
            delta_pen = self.lambda_risk * (notionals[i] ** 2) * 1e-14

            # Analytical P(hit) across grid
            bp = max(min(base_probs[i], 0.999), 0.001)
            base_logit = np.log(bp / (1 - bp))
            logits = base_logit + spread_coef * (spread_grid - base_spreads[i])
            probs = 1.0 / (1.0 + np.exp(-logits))
            epnls = probs * (spread_grid - hedge_cost) - delta_pen

            best_idx = np.argmax(epnls)
            opt_spreads[i] = spread_grid[best_idx]
            opt_epnls[i] = epnls[best_idx]
            opt_probs[i] = probs[best_idx]

            if opt_spreads[i] < self.MIN_SPREAD_PIPS:
                guardrail_count += 1

        realized_pnls = np.where(was_hits,
                                  opt_spreads - np.array([self.HEDGE_COST_PIPS.get(p, 0.15) for p in pairs]),
                                  0.0)

        return {
            "total_expected_pnl": round(float(opt_epnls.sum()), 2),
            "total_realized_pnl": round(float(realized_pnls.sum()), 2),
            "avg_spread": round(float(opt_spreads.mean()), 3),
            "avg_hit_prob": round(float(opt_probs.mean()), 4),
            "hit_rate_actual": round(float(was_hits.mean()), 4),
            "n_rfqs": n,
            "n_guardrails_triggered": guardrail_count,
        }

    # ------------------------------------------------------------------
    # Flat spread comparison
    # ------------------------------------------------------------------

    def quote_vs_flat(self, rfq_data: pd.DataFrame,
                      flat_spread_pips: float = 1.0) -> pd.DataFrame:
        """Compare optimizer quotes vs a flat (constant) spread strategy.

        Example output:
            strategy   | total_epnl | avg_spread | avg_hit_prob | n_rfqs
            optimizer  | 4250.5     | 0.92       | 0.68         | 10000
            flat_1.0   | 3100.2     | 1.00       | 0.62         | 10000
            improvement| +1150.3    | -0.08      | +0.06        |

        Parameters
        ----------
        rfq_data : pd.DataFrame
            Historical RFQ data.
        flat_spread_pips : float
            The constant spread for the benchmark strategy.

        Returns
        -------
        pd.DataFrame
            Comparison table between optimizer and flat strategy.
        """
        # Optimizer backtest
        opt_bt = self.backtest(rfq_data)

        # Flat spread backtest — single batch predict
        flat_data = rfq_data.copy()
        flat_data["quoted_spread_pips"] = flat_spread_pips
        try:
            flat_probs = self.win_model.predict(flat_data)
        except Exception:
            flat_probs = 1.0 / (1.0 + np.exp(1.2 * (flat_spread_pips - 1.5))) * np.ones(len(rfq_data))

        pairs = rfq_data["pair"].values if "pair" in rfq_data.columns else np.full(len(rfq_data), "EUR/USD")
        was_hits = rfq_data["was_hit"].values if "was_hit" in rfq_data.columns else np.zeros(len(rfq_data), dtype=bool)
        hedge_costs = np.array([self.HEDGE_COST_PIPS.get(p, 0.15) for p in pairs])

        flat_epnl = flat_probs * (flat_spread_pips - hedge_costs)
        flat_realized = np.where(was_hits, flat_spread_pips - hedge_costs, 0.0)

        comparison = pd.DataFrame([
            {
                "strategy": "optimizer",
                "total_expected_pnl": opt_bt["total_expected_pnl"],
                "total_realized_pnl": opt_bt.get("total_realized_pnl", 0),
                "avg_spread_pips": opt_bt.get("avg_spread", opt_bt.get("avg_spread_pips", 0)),
                "avg_hit_prob": opt_bt["avg_hit_prob"],
                "n_rfqs": opt_bt["n_rfqs"],
            },
            {
                "strategy": f"flat_{flat_spread_pips}",
                "total_expected_pnl": round(float(flat_epnl.sum()), 2),
                "total_realized_pnl": round(float(flat_realized.sum()), 2),
                "avg_spread_pips": flat_spread_pips,
                "avg_hit_prob": round(float(flat_probs.mean()), 4),
                "n_rfqs": len(rfq_data),
            },
        ])

        # Add improvement row
        improvement_pips = comparison.iloc[0]["total_expected_pnl"] - comparison.iloc[1]["total_expected_pnl"]
        imp = {
            "strategy": "improvement",
            "total_expected_pnl": round(improvement_pips, 2),
            "total_realized_pnl": round(
                comparison.iloc[0]["total_realized_pnl"] - comparison.iloc[1]["total_realized_pnl"], 2
            ),
            "avg_spread_pips": round(
                comparison.iloc[0]["avg_spread_pips"] - comparison.iloc[1]["avg_spread_pips"], 3
            ),
            "avg_hit_prob": round(
                comparison.iloc[0]["avg_hit_prob"] - comparison.iloc[1]["avg_hit_prob"], 4
            ),
            "n_rfqs": len(rfq_data),
        }
        comparison = pd.concat([comparison, pd.DataFrame([imp])], ignore_index=True)

        return comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fix_adjustment(self, fix_sched: Dict, alpha_skew_pips: float) -> float:
        """Compute the fix-timing adjustment to expected P&L.

        Pre-fix: positive adjustment (signal is profitable, tighten spread)
        At-fix: negative adjustment (adverse selection, widen)
        Post-fix: slight positive (reversal opportunity)

        Example:
            Pre-fix, 15 min out, alpha_skew = +1.2 pips
            fix_adj = 0.05 + 0.3 * 1.2 * (1 - 15/60) = 0.05 + 0.27 = 0.32

        Parameters
        ----------
        fix_sched : dict
            From FixAlphaModel.fix_schedule().
        alpha_skew_pips : float
            Alpha skew signal in pips.

        Returns
        -------
        float
            P&L adjustment in pips.
        """
        proximity = fix_sched.get("fix_proximity", "neutral")
        minutes = fix_sched.get("minutes_to_fix", 999)

        if proximity == "pre_fix":
            # Pre-fix: we benefit from the drift, so tighten aggressively
            time_factor = max(0, 1.0 - minutes / 60.0)
            return 0.05 + 0.3 * abs(alpha_skew_pips) * time_factor

        elif proximity == "at_fix":
            # At fix: adverse selection is highest
            return -0.2 * fix_sched.get("adverse_selection_risk", 0.8)

        elif proximity == "post_fix":
            # Post-fix: reversal opportunity, slight benefit
            return 0.02

        else:
            return 0.0

    def _check_guardrails(self, spread: float, notional: float,
                          hit_prob: float, fix_sched: Dict) -> list:
        """Check and return triggered guardrails.

        Example:
            spread=0.05, notional=80MM, fix_proximity='at_fix', prob=0.03
            -> ['min_spread_breach', 'high_notional_warning',
                'fix_window_widening', 'low_probability_reject']
        """
        triggered = []

        if spread < self.MIN_SPREAD_PIPS:
            triggered.append("min_spread_breach")

        if notional > self.HIGH_NOTIONAL_THRESHOLD:
            triggered.append("high_notional_warning")

        if fix_sched.get("fix_proximity") == "at_fix":
            triggered.append("fix_window_widening")

        if hit_prob < 0.05:
            triggered.append("low_probability_reject")

        return triggered
