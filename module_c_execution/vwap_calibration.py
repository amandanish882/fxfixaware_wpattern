"""Calibrate intraday VWAP volume profile from cached CME MBP-10 deep-book data.

Computes a 96-bucket (15-min) volume profile by aggregating bid+ask size at
L1-L3 across all cached MBP-10 days. Real-data buckets cover 13:00-17:00 UTC
(the London-NY overlap, where the MBP-10 cache lives); buckets outside that
window are filled with the legacy expert curve and the result is renormalized
so the profile integrates to 1.0 across 96 buckets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "databento_cache"

# Legacy expert curve to use OUTSIDE the 13:00-17:00 UTC real-data window.
# Same shape as the 24-element scheduler default so non-overlap buckets
# stay continuous with the historical baseline.
_EXPERT_24H = [
    # 00:00 - 06:00 (Asia tail, low)
    0.02, 0.02, 0.02, 0.02, 0.03, 0.04,
    # 06:00 - 12:00 (London open, building)
    0.06, 0.10, 0.14, 0.18, 0.20, 0.22,
    # 12:00 - 18:00 (London-NY peak)
    0.30, 0.50, 0.80, 1.00, 0.95, 0.70,
    # 18:00 - 24:00 (NY fade, Asia open)
    0.40, 0.20, 0.10, 0.06, 0.04, 0.03,
]


def _build_expert_15m_profile() -> np.ndarray:
    """Linear-interpolate the 24-hour expert curve to 96 fifteen-minute buckets."""
    hours = np.arange(24)
    target = np.linspace(0, 23, 96)
    interp = np.interp(target, hours, _EXPERT_24H)
    return interp / interp.sum()


def calibrate_vwap_profile() -> np.ndarray:
    """Return a 96-bucket 15-min VWAP volume profile (sums to 1.0).

    Real-data calibration over 13:00-17:00 UTC (buckets 52-67) using all
    cached MBP-10 parquets; legacy expert curve elsewhere; final renormalised
    across the 96 buckets.
    """
    expert = _build_expert_15m_profile()  # 96 weights, sums to 1

    files = sorted(CACHE_DIR.glob("mbp10_fx_*.parquet"))
    if not files:
        return expert

    bucket_volume = np.zeros(96)
    depth_candidates = [
        "bid_sz_00", "ask_sz_00",
        "bid_sz_01", "ask_sz_01",
        "bid_sz_02", "ask_sz_02",
    ]

    for f in files:
        try:
            df = pd.read_parquet(
                f,
                columns=["ts_event"] + depth_candidates,
            )
        except Exception:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
        if df.empty or "ts_event" not in df.columns:
            continue
        ts = pd.to_datetime(df["ts_event"], utc=True)
        bucket_idx = ts.dt.hour.values * 4 + ts.dt.minute.values // 15

        depth_cols = [c for c in depth_candidates if c in df.columns]
        if not depth_cols:
            continue
        total_depth = df[depth_cols].sum(axis=1).values
        # Aggregate per bucket via numpy bincount (vectorised)
        valid = ~np.isnan(total_depth) & (bucket_idx >= 0) & (bucket_idx < 96)
        if not valid.any():
            continue
        bucket_volume += np.bincount(
            bucket_idx[valid].astype(np.int64),
            weights=total_depth[valid].astype(np.float64),
            minlength=96,
        )

    if bucket_volume.sum() <= 0:
        return expert

    # Real-data buckets: 13:00 = bucket 52, 17:00 = bucket 68
    real_lo, real_hi = 52, 68
    real_block = bucket_volume[real_lo:real_hi]
    if real_block.sum() <= 0:
        return expert

    # Build merged profile: real where we have it, expert elsewhere.
    merged = expert.copy()
    expert_block_total = expert[real_lo:real_hi].sum()
    if real_block.sum() > 0 and expert_block_total > 0:
        scaled_real = real_block / real_block.sum() * expert_block_total
        merged[real_lo:real_hi] = scaled_real

    # Renormalize so 96 buckets sum to 1
    s = merged.sum()
    if s > 0:
        merged = merged / s
    return merged


def calibrate_vwap_profile_24h() -> np.ndarray:
    """24-hour version: aggregate the 96-bucket profile down to hourly weights.

    Returns a length-24 array that sums to 1.0, suitable as a drop-in
    replacement for the legacy hourly profile in ``execution_scheduler``.
    """
    p = calibrate_vwap_profile()
    return p.reshape(24, 4).sum(axis=1)
