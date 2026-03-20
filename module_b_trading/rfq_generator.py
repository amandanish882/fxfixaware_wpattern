"""
FX RFQ (Request-for-Quote) Generator
=====================================

Generates realistic synthetic FX RFQ flow for backtesting the market-making
system. Calibrated to realistic G4 FX market parameters.

Flow characteristics:
    - Pairs: EUR/USD (40%), GBP/USD (25%), JPY/USD (20%), AUD/USD (15%)
    - Client segments: hedge_fund (25%), real_money (30%), corporate (35%), central_bank (10%)
    - Notionals: lognormal in [1MM, 100MM] USD
    - Spreads: 0.3-3.0 pips for majors
    - Business hours: London 7:00-17:00 UTC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, Optional, List

from .fix_alpha_signals import FixAlphaModel, DEFAULT_FIX_TIMES


# Distribution weights
PAIR_WEIGHTS = {"EUR/USD": 0.40, "GBP/USD": 0.25, "JPY/USD": 0.20, "AUD/USD": 0.15}
SEGMENT_WEIGHTS = {"hedge_fund": 0.25, "real_money": 0.30, "corporate": 0.35, "central_bank": 0.10}

# Default spot rates (mid-2024 levels)
DEFAULT_FX_SPOTS = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2650,
    "JPY/USD": 149.50,
    "AUD/USD": 0.6550,
}


class FXRFQGenerator:
    """Generate synthetic FX RFQ flow for backtesting.

    Example usage:
        gen = FXRFQGenerator()
        rfqs = gen.generate(
            n_rfqs=10_000,
            fx_spots={"EUR/USD": 1.0850, "GBP/USD": 1.2650},
            start_date="2024-01-02",
            end_date="2024-12-31",
        )
        # rfqs is a DataFrame with ~10,000 rows

    Example output row:
        timestamp          | client_id | client_segment | pair    | direction  | notional_usd | spot_mid | quoted_spread_pips | was_hit | fix_proximity
        2024-03-15 14:30   | C_0042    | corporate      | EUR/USD | buy_base   | 5,200,000    | 1.0855   | 0.82               | True    | pre_fix
    """

    def __init__(self):
        self.fix_model = FixAlphaModel()

    def generate(
        self,
        n_rfqs: int,
        fx_spots: Optional[Dict[str, float]] = None,
        start_date: str = "2024-01-02",
        end_date: str = "2024-12-31",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate n_rfqs synthetic FX RFQ records.

        Parameters
        ----------
        n_rfqs : int
            Number of RFQs to generate. E.g. 10,000 for ~40 RFQs per business day.
        fx_spots : dict, optional
            Spot rates per pair. Defaults to mid-2024 levels.
        start_date : str
            Start date string 'YYYY-MM-DD'.
        end_date : str
            End date string 'YYYY-MM-DD'.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, client_id, client_segment, pair, direction,
            notional_usd, spot_mid, quoted_spread_pips, was_hit, fix_proximity.
        """
        rng = np.random.default_rng(seed)
        spots = fx_spots or dict(DEFAULT_FX_SPOTS)

        # Generate business day timestamps
        timestamps = self._generate_timestamps(n_rfqs, start_date, end_date, rng)

        # Pair selection
        pair_names = list(PAIR_WEIGHTS.keys())
        pair_probs = np.array([PAIR_WEIGHTS[p] for p in pair_names])
        pairs = rng.choice(pair_names, size=n_rfqs, p=pair_probs)

        # Client segments
        seg_names = list(SEGMENT_WEIGHTS.keys())
        seg_probs = np.array([SEGMENT_WEIGHTS[s] for s in seg_names])
        segments = rng.choice(seg_names, size=n_rfqs, p=seg_probs)

        # Client IDs (200 unique clients)
        n_clients = 200
        client_ids = np.array([f"C_{i:04d}" for i in range(n_clients)])
        client_id_arr = rng.choice(client_ids, size=n_rfqs)

        # Direction: roughly 50/50 with slight client-segment bias
        directions = self._generate_directions(n_rfqs, segments, rng)

        # Notionals: lognormal in [1MM, 100MM] USD
        # lognormal with mu=16.5 (~14.5MM median), sigma=0.8
        raw_notional = rng.lognormal(mean=16.5, sigma=0.8, size=n_rfqs)
        notionals = np.clip(raw_notional, 1_000_000, 100_000_000)
        # Round to nearest 100K
        notionals = np.round(notionals / 100_000) * 100_000

        # Spot mid: base spot + small random walk
        spot_mids = np.array([spots.get(p, 1.0) for p in pairs])
        # Add small noise: ±20 pips for most pairs, ±200 pips for JPY
        for i in range(n_rfqs):
            pip_size = 0.01 if "JPY" in pairs[i] else 0.0001
            noise_pips = rng.normal(0, 10)  # ±10 pips std
            spot_mids[i] += noise_pips * pip_size

        # Quoted spread: uniform [0.3, 3.0] pips, adjusted by segment and notional
        base_spread = rng.uniform(0.3, 3.0, size=n_rfqs)
        spreads = self._adjust_spreads(base_spread, segments, notionals, rng)

        # Fix proximity
        fix_proximities = np.array([
            self._compute_fix_proximity(ts) for ts in timestamps
        ])

        # Was hit: logistic model
        was_hit = self._compute_hit_probability(
            spreads, notionals, segments, pairs, fix_proximities, rng
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "client_id": client_id_arr,
            "client_segment": segments,
            "pair": pairs,
            "direction": directions,
            "notional_usd": notionals,
            "spot_mid": np.round(spot_mids, 5),
            "quoted_spread_pips": np.round(spreads, 2),
            "was_hit": was_hit,
            "fix_proximity": fix_proximities,
        })

        return df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Timestamp generation
    # ------------------------------------------------------------------

    def _generate_timestamps(
        self, n: int, start_date: str, end_date: str, rng: np.random.Generator
    ) -> List[datetime]:
        """Generate n timestamps on business days during London hours (7-17 UTC).

        Example: 2024-03-15 14:32:17 UTC (a Friday during London/NY overlap)
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Generate business dates
        biz_dates = pd.bdate_range(start, end)
        if len(biz_dates) == 0:
            biz_dates = pd.bdate_range(start, start + pd.Timedelta(days=365))

        # Pick random dates (with replacement)
        date_indices = rng.integers(0, len(biz_dates), size=n)

        timestamps = []
        for idx in date_indices:
            d = biz_dates[idx].to_pydatetime()
            # Random time between 7:00 and 17:00 UTC (600 minutes)
            minutes_offset = rng.integers(0, 600)
            hour = 7 + minutes_offset // 60
            minute = minutes_offset % 60
            second = rng.integers(0, 60)
            ts = d.replace(hour=hour, minute=minute, second=second)
            timestamps.append(ts)

        return timestamps

    # ------------------------------------------------------------------
    # Direction generation
    # ------------------------------------------------------------------

    def _generate_directions(
        self, n: int, segments: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate buy/sell directions with slight segment bias.

        Corporates lean towards hedging (buy_base for importers),
        hedge funds are balanced, central banks lean towards selling.
        """
        buy_prob_by_segment = {
            "hedge_fund": 0.50,
            "real_money": 0.55,
            "corporate": 0.60,
            "central_bank": 0.40,
        }
        probs = np.array([buy_prob_by_segment.get(s, 0.5) for s in segments])
        is_buy = rng.random(n) < probs
        return np.where(is_buy, "buy_base", "sell_base")

    # ------------------------------------------------------------------
    # Spread adjustment
    # ------------------------------------------------------------------

    def _adjust_spreads(
        self,
        base_spread: np.ndarray,
        segments: np.ndarray,
        notionals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Adjust spreads by client segment and notional.

        Hedge funds get tighter spreads (more price-sensitive),
        corporates get wider (less price-sensitive),
        central banks get tightest (relationship + huge flow).
        Large notionals get slightly wider spreads (more risk).

        Example:
            Base spread = 1.0 pips, hedge_fund, 5MM USD
            -> 1.0 * 0.80 * (1 + 0.02 * log(5/1)) = 0.80 * 1.032 = 0.826 pips
        """
        segment_multiplier = {
            "hedge_fund": 0.80,
            "real_money": 1.00,
            "corporate": 1.20,
            "central_bank": 0.70,
        }
        mults = np.array([segment_multiplier.get(s, 1.0) for s in segments])

        # Notional impact: wider for larger notionals (log scale)
        notional_factor = 1.0 + 0.02 * np.log(notionals / 1_000_000)

        adjusted = base_spread * mults * notional_factor
        return np.clip(adjusted, 0.1, 5.0)

    # ------------------------------------------------------------------
    # Fix proximity
    # ------------------------------------------------------------------

    def _compute_fix_proximity(self, ts: datetime) -> str:
        """Classify timestamp relative to fix windows.

        Example:
            15:50 UTC -> 'pre_fix' (10 min before WMR 16:00)
            16:02 UTC -> 'at_fix' (2 min after WMR)
            16:20 UTC -> 'post_fix' (20 min after WMR)
            10:00 UTC -> 'neutral'
        """
        sched = self.fix_model.fix_schedule(ts)
        return sched["fix_proximity"]

    # ------------------------------------------------------------------
    # Hit probability (logistic model)
    # ------------------------------------------------------------------

    def _compute_hit_probability(
        self,
        spreads: np.ndarray,
        notionals: np.ndarray,
        segments: np.ndarray,
        pairs: np.ndarray,
        fix_proximities: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Logistic hit probability model.

        P(hit) = sigmoid(intercept - beta_spread * spread + beta_segment + ...)

        Example:
            spread = 0.5 pips, hedge_fund, EUR/USD, neutral
            logit = 1.5 - 1.2*0.5 + 0.3 + 0.1 = 1.3
            P(hit) = sigmoid(1.3) = 0.786 -> ~79% hit rate

            spread = 2.5 pips, corporate, AUD/USD, neutral
            logit = 1.5 - 1.2*2.5 - 0.2 - 0.1 = -1.8
            P(hit) = sigmoid(-1.8) = 0.142 -> ~14% hit rate
        """
        intercept = 1.5

        # Spread: wider spread -> lower hit probability
        beta_spread = -1.2

        # Segment effect
        segment_betas = {
            "hedge_fund": 0.3,    # very price-sensitive, hit more at tight spreads
            "real_money": 0.0,
            "corporate": -0.2,    # less price-sensitive
            "central_bank": 0.5,  # relationship-driven, high fill rate
        }
        seg_effect = np.array([segment_betas.get(s, 0.0) for s in segments])

        # Pair effect (EUR/USD most liquid -> higher hit)
        pair_betas = {"EUR/USD": 0.1, "GBP/USD": 0.0, "JPY/USD": -0.1, "AUD/USD": -0.1}
        pair_effect = np.array([pair_betas.get(p, 0.0) for p in pairs])

        # Notional effect: larger notionals slightly less likely to hit
        notional_effect = -0.1 * np.log(notionals / 5_000_000)

        # Fix proximity effect: at_fix -> slightly lower hit (wider spread environment)
        fix_effect = np.array([
            -0.3 if fp == "at_fix" else (0.1 if fp == "pre_fix" else 0.0)
            for fp in fix_proximities
        ])

        logit = intercept + beta_spread * spreads + seg_effect + pair_effect + notional_effect + fix_effect

        prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        was_hit = rng.random(len(prob)) < prob
        return was_hit
