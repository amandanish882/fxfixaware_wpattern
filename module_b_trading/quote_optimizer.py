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
        """Find the optimal asymmetric quote for an RFQ.

        E[PnL] objective:
            E[PnL] = P(hit|s) * (s - hedge_cost + alpha_per_trade)
                    - lambda * delta^2 + regime_boost

        where alpha_per_trade = (signed) alpha_skew_pips * proximity_factor,
        oriented to the RFQ direction so positive = filling is alpha-favorable.

        Convention for ``alpha_skew_pips``:
            Signed directional view in pips (positive = USD up = base down).
            Per-RFQ favorability is derived from the RFQ ``direction`` field:
            ``buy_base`` (client takes base, gives us USD) is favorable when
            alpha > 0; ``sell_base`` is favorable when alpha < 0.

        Returns
        -------
        dict with:
            optimal_spread_pips     : chosen half-spread (pips). Already
                                      direction-adjusted via per-trade alpha:
                                      tighter on alpha-favorable RFQs,
                                      wider on alpha-adverse.
            expected_pnl            : E[PnL] at the chosen spread (pips)
            expected_alpha_pnl_pips : alpha contribution to E[PnL] (pips)
            hit_prob                : P(hit) at chosen spread
            guardrails              : list of triggered guardrails
            fix_schedule            : fix-timing dict
            hedge_cost_pips         : assumed hedging cost
            alpha_skew_pips         : echoed input directional view
            alpha_per_trade_pips    : proximity-scaled, RFQ-direction-signed
                                      per-trade alpha (pips)
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

        # Per-RFQ favorability: flip sign for sell_base (the client gives us base,
        # so a "USD up" view is unfavorable for accumulating that direction).
        rfq_direction = str(row.get("direction", "buy_base"))
        rfq_sign = +1.0 if rfq_direction == "buy_base" else -1.0
        proximity_factor = self._alpha_proximity_factor(fix_sched)
        alpha_per_trade = alpha_skew_pips * rfq_sign * proximity_factor

        # Regime boost: small constant edge in pre/post-fix; widening cost at-fix
        regime_boost = self._regime_boost(fix_sched)

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

        # Vectorized E[PnL]: alpha enters per-fill revenue (multiplied by hit prob)
        epnls = probs * (spread_grid - hedge_cost + alpha_per_trade) - delta_penalty + regime_boost

        best_idx = np.argmax(epnls)
        best_spread = spread_grid[best_idx]
        best_epnl = epnls[best_idx]
        best_prob = probs[best_idx]

        # Apply guardrails
        guardrails = self._check_guardrails(best_spread, notional, best_prob, fix_sched)
        if "fix_window_widening" in guardrails:
            best_spread = max(best_spread, best_spread * fix_sched["spread_multiplier"])
        if "min_spread_breach" in guardrails:
            best_spread = max(best_spread, self.MIN_SPREAD_PIPS)

        return {
            "optimal_spread_pips": round(float(best_spread), 3),
            "expected_pnl": round(float(best_epnl), 4),
            "expected_alpha_pnl_pips": round(float(alpha_per_trade * best_prob), 4),
            "hit_prob": round(float(best_prob), 4),
            "guardrails": guardrails,
            "fix_schedule": fix_sched,
            "hedge_cost_pips": hedge_cost,
            "alpha_skew_pips": round(float(alpha_skew_pips), 3),
            "alpha_per_trade_pips": round(float(alpha_per_trade), 4),
        }

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        rfq_data: pd.DataFrame,
        alpha_skew_pips: Optional[Any] = None,
        realized_alpha_pips: Optional[Any] = "same",
    ) -> Dict:
        """Backtest the optimizer across a historical RFQ dataset.

        Parameters
        ----------
        rfq_data : pd.DataFrame
            Historical RFQ data with ``was_hit`` column. If ``timestamp`` and
            ``direction`` columns are present they are used for fix-proximity
            scaling and per-RFQ alpha sign.
        alpha_skew_pips : float, np.ndarray, pd.Series, or None
            Directional alpha view (in pips) the OPTIMIZER SEES — drives the
            spread choice. ``None`` means the optimizer is alpha-blind.
        realized_alpha_pips : same shapes, or ``"same"``, or ``None``
            Directional alpha view used to EVALUATE the chosen spreads —
            represents the *true* drift the world delivers. Default ``"same"``
            mirrors the optimizer's view (the standard backtest). Pass the
            true alpha here while leaving ``alpha_skew_pips=None`` to compute
            "what an alpha-blind optimizer actually achieves under true drift",
            i.e. the baseline for value-of-information lift.

        Returns
        -------
        dict
            Backtest summary, including:
              - ``total_expected_pnl`` evaluated under realized alpha
              - ``total_alpha_pnl`` directional contribution under realized alpha
              - ``opt_spreads_per_rfq`` array of per-RFQ chosen spreads
        """
        n = len(rfq_data)
        if n == 0:
            return {"total_expected_pnl": 0, "total_realized_pnl": 0,
                    "total_alpha_pnl": 0,
                    "avg_spread": 0,
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
        directions = rfq_data["direction"].values if "direction" in rfq_data.columns else np.full(n, "buy_base")
        timestamps = rfq_data["timestamp"].values if "timestamp" in rfq_data.columns else np.array([None] * n)

        def _broadcast(x):
            if x is None:
                return np.zeros(n)
            if np.isscalar(x):
                return np.full(n, float(x))
            arr = np.asarray(x, dtype=float)
            return np.full(n, float(arr.item())) if arr.size == 1 else arr

        # The optimizer SEES this alpha when picking spreads.
        alpha_arr = _broadcast(alpha_skew_pips)
        # The world DELIVERS this alpha when evaluating outcomes. Default = same.
        if isinstance(realized_alpha_pips, str) and realized_alpha_pips == "same":
            realized_arr = alpha_arr.copy()
        else:
            realized_arr = _broadcast(realized_alpha_pips)

        rfq_signs = np.where(directions == "buy_base", 1.0, -1.0)

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
        opt_alpha_epnls = np.zeros(n)
        opt_probs = np.zeros(n)
        guardrail_count = 0

        for i in range(n):
            hedge_cost = self.HEDGE_COST_PIPS.get(pairs[i], 0.15)
            delta_pen = self.lambda_risk * (notionals[i] ** 2) * 1e-14

            # Fix schedule + proximity factor for this RFQ
            ts = timestamps[i]
            if ts is None:
                fix_sched = {"fix_proximity": "neutral", "minutes_to_fix": 999,
                             "spread_multiplier": 1.0, "adverse_selection_risk": 0.1}
            else:
                if not isinstance(ts, datetime):
                    ts = pd.Timestamp(ts).to_pydatetime()
                fix_sched = self.fix_alpha.fix_schedule(ts)
            proximity_factor = self._alpha_proximity_factor(fix_sched)
            regime_boost = self._regime_boost(fix_sched)

            chosen_alpha = alpha_arr[i] * rfq_signs[i] * proximity_factor
            realized_alpha = realized_arr[i] * rfq_signs[i] * proximity_factor

            # Analytical P(hit) across grid
            bp = max(min(base_probs[i], 0.999), 0.001)
            base_logit = np.log(bp / (1 - bp))
            logits = base_logit + spread_coef * (spread_grid - base_spreads[i])
            probs = 1.0 / (1.0 + np.exp(-logits))
            # Optimizer picks spread under the alpha it CAN SEE
            epnls_chosen_view = probs * (spread_grid - hedge_cost + chosen_alpha) - delta_pen + regime_boost

            best_idx = int(np.argmax(epnls_chosen_view))
            best_spread = float(spread_grid[best_idx])
            best_prob = float(probs[best_idx])

            # Guardrails
            if fix_sched.get("fix_proximity") == "at_fix":
                best_spread = max(best_spread, best_spread * fix_sched.get("spread_multiplier", 1.0))
                guardrail_count += 1
            if best_spread < self.MIN_SPREAD_PIPS:
                best_spread = self.MIN_SPREAD_PIPS

            # Re-evaluate the chosen spread against the *realized* alpha for the
            # honest E[PnL]. (When chosen=realized this collapses to the same
            # number; when they differ this exposes information value.)
            best_logit = base_logit + spread_coef * (best_spread - base_spreads[i])
            best_prob_at_chosen = 1.0 / (1.0 + np.exp(-best_logit))
            true_epnl = best_prob_at_chosen * (best_spread - hedge_cost + realized_alpha) - delta_pen + regime_boost

            opt_spreads[i] = best_spread
            opt_epnls[i] = float(true_epnl)
            opt_alpha_epnls[i] = float(best_prob_at_chosen * realized_alpha)
            opt_probs[i] = float(best_prob_at_chosen)

        # Realized PnL: when a fill happens, the trader captures the quoted
        # spread minus hedge cost, plus the realized alpha drift.
        hedge_costs_arr = np.array([self.HEDGE_COST_PIPS.get(p, 0.15) for p in pairs])
        proximity_factors = np.array([
            self._alpha_proximity_factor(
                self.fix_alpha.fix_schedule(
                    pd.Timestamp(t).to_pydatetime() if t is not None else datetime(2000, 1, 1)
                )
            ) for t in timestamps
        ])
        realized_pnls = np.where(
            was_hits,
            opt_spreads - hedge_costs_arr + realized_arr * rfq_signs * proximity_factors,
            0.0,
        )

        return {
            "total_expected_pnl": round(float(opt_epnls.sum()), 2),
            "total_realized_pnl": round(float(realized_pnls.sum()), 2),
            "total_alpha_pnl": round(float(opt_alpha_epnls.sum()), 2),
            "avg_spread": round(float(opt_spreads.mean()), 3),
            "avg_hit_prob": round(float(opt_probs.mean()), 4),
            "hit_rate_actual": round(float(was_hits.mean()), 4),
            "n_rfqs": n,
            "n_guardrails_triggered": guardrail_count,
            "opt_spreads_per_rfq": opt_spreads.copy(),
            "opt_alpha_epnls_per_rfq": opt_alpha_epnls.copy(),
        }

    # ------------------------------------------------------------------
    # Flat spread comparison
    # ------------------------------------------------------------------

    def quote_vs_flat(self, rfq_data: pd.DataFrame,
                      flat_spread_pips: float = 1.0,
                      alpha_skew_pips: Optional[Any] = None) -> pd.DataFrame:
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
        # Optimizer backtest (with directional alpha threaded in)
        opt_bt = self.backtest(rfq_data, alpha_skew_pips=alpha_skew_pips)

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

    def _regime_boost(self, fix_sched: Dict) -> float:
        """Constant E[PnL] adjustment from being inside a fix regime.

        Captures the non-alpha edge available to a fix-aware market maker:
        a small bonus for being open during the alpha window (pre/post fix)
        and a penalty for adverse selection at the fix print.
        """
        proximity = fix_sched.get("fix_proximity", "neutral")
        minutes = fix_sched.get("minutes_to_fix", 999)

        if proximity == "pre_fix":
            time_factor = max(0.0, 1.0 - minutes / 60.0)
            return 0.05 * time_factor
        if proximity == "at_fix":
            return -0.2 * fix_sched.get("adverse_selection_risk", 0.8)
        if proximity == "post_fix":
            return 0.02
        return 0.0

    def _alpha_proximity_factor(self, fix_sched: Dict) -> float:
        """How much of the alpha signal flows into per-trade E[PnL].

        Strongest near the fix (signal is most reliable when concentrated
        flow is imminent), suppressed at the fix print (adverse selection
        dominates), small but non-zero outside fix windows.
        """
        proximity = fix_sched.get("fix_proximity", "neutral")
        minutes = fix_sched.get("minutes_to_fix", 999)

        if proximity == "pre_fix":
            return max(0.3, 1.0 - minutes / 60.0)  # 0.3 -> 1.0 ramp
        if proximity == "at_fix":
            return 0.0  # adverse selection dominates; ignore directional view
        if proximity == "post_fix":
            return 0.5  # reversal partially captured
        return 0.3  # always-on baseline (carry/momentum/MR still apply)

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
