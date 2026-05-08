"""Pure basis composite alpha — realistic backtest with toxic flow filter.

Fixes modeled:
  1. Accumulated inventory: λ × (net_position + fill)²
  2. Correlated adverse selection: daily market shock on ALL fills
  3. Adverse selection: per-fill, calibrated by client segment (hedge funds
     more toxic than corporates), fix proximity (at-fix most toxic), and
     notional (large = more informed)
  4. Inventory mark-to-market + EOD flatten cost
  5. **Toxic flow filter**: logistic regression trained on first 50% of RFQs
     to predict P(toxic | features). RFQs with P(toxic) > threshold are
     rejected (last-look) or quoted prohibitively wide.

Composite = 50% basis_fix + 25% basis_carry + 15% basis_momentum + 10% basis_MR
"""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from module_b_trading.fix_alpha_signals import FixAlphaModel
from module_b_trading.quote_optimizer import FXQuoteOptimizer
from module_b_trading.rfq_generator import FXRFQGenerator
from shared.plot_style import JPM_COLORS, set_jpm_style

PROJECT_ROOT = Path(__file__).resolve().parent
BASIS_CACHE = PROJECT_ROOT / "data" / "cme_15m_basis_cache.parquet"
G4 = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]
FX_SPOTS = {"EUR/USD": 1.1715, "GBP/USD": 1.3513, "JPY/USD": 0.00627, "AUD/USD": 0.7181}
LONDON_FIX_BUCKET = 64
MAX_SKEW_PIPS = 3.0

L2_HEDGE_COST = {"EUR/USD": 0.15, "GBP/USD": 0.25, "JPY/USD": 0.20, "AUD/USD": 0.30}
DAILY_VOL = {"EUR/USD": 55, "GBP/USD": 75, "JPY/USD": 60, "AUD/USD": 85}
EOD_FLATTEN_COST = {"EUR/USD": 0.30, "GBP/USD": 0.50, "JPY/USD": 0.40, "AUD/USD": 0.60}

# Adverse selection calibration by client segment (mean pips, negative = against us)
ADVERSE_BY_SEGMENT = {
    "hedge_fund": -1.8,     # most toxic — HFT / informed flow
    "real_money": -0.6,     # moderate — pension / insurance rebalance
    "corporate": -0.3,      # least toxic — hedging, uninformed
    "central_bank": -0.2,   # relationship flow, low toxicity
    "retail": -0.8,         # aggregated retail, moderate
}
ADVERSE_BY_FIX = {
    "at_fix": -1.5,         # fix window = peak adverse selection
    "pre_fix": -0.5,        # building to fix
    "post_fix": -0.3,       # post-fix reversal
    "neutral": 0.0,
}


class RealisticWinModel:
    def __init__(self, steepness=-4.0, midpoint=0.6):
        self.steepness = steepness
        self.midpoint = midpoint
        self.model = None
        self.feature_columns = None

    def predict(self, rfq_data):
        s = rfq_data["quoted_spread_pips"].values if "quoted_spread_pips" in rfq_data.columns else np.ones(len(rfq_data))
        return 1.0 / (1.0 + np.exp(-self.steepness * (s - self.midpoint)))

    def prepare_features(self, d):
        return d

    def train(self, d):
        return {}


# ── Basis signals ────────────────────────────────────────────────────

def build_basis_signals():
    df = pd.read_parquet(BASIS_CACHE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["bucket"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15

    fix_rows = []
    for (date, pair), g in df.groupby(["date", "pair"]):
        pre = g[(g["bucket"] >= LONDON_FIX_BUCKET - 2) & (g["bucket"] < LONDON_FIX_BUCKET)]
        fix_rows.append({"date": date, "pair": pair,
                         "pre_fix_basis": pre["basis_change_bps"].sum() if len(pre) else 0})
    fix_df = pd.DataFrame(fix_rows).sort_values(["pair", "date"])
    fix_df["fix_signal_raw"] = fix_df.groupby("pair")["pre_fix_basis"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    q = max(fix_df["fix_signal_raw"].abs().quantile(0.95), 0.01)
    fix_df["fix_signal"] = np.clip(-fix_df["fix_signal_raw"] / q, -1, 1)

    eod = df.groupby(["date", "pair"])["log_basis_bps"].last().reset_index().rename(columns={"log_basis_bps": "eod_basis"})
    eod = eod.sort_values(["pair", "date"])
    med = eod.groupby("date")["eod_basis"].transform("median")
    std = eod.groupby("date")["eod_basis"].transform("std").clip(lower=0.01)
    eod["carry_signal"] = np.clip((eod["eod_basis"] - med) / std, -1, 1)

    dc = df.groupby(["date", "pair"])["basis_change_bps"].sum().reset_index().rename(columns={"basis_change_bps": "daily_chg"})
    dc = dc.sort_values(["pair", "date"])
    dc["ema5"] = dc.groupby("pair")["daily_chg"].transform(lambda x: x.ewm(span=5, min_periods=3).mean())
    q2 = max(dc["ema5"].abs().quantile(0.95), 0.01)
    dc["momentum_signal"] = np.clip(dc["ema5"] / q2, -1, 1)

    eod["ma20"] = eod.groupby("pair")["eod_basis"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    eod["std20"] = eod.groupby("pair")["eod_basis"].transform(lambda x: x.rolling(20, min_periods=10).std()).clip(lower=0.01)
    eod["mr_signal"] = np.clip(-(eod["eod_basis"] - eod["ma20"]) / eod["std20"] / 2, -1, 1)

    out = fix_df[["date", "pair", "fix_signal"]].copy()
    out = out.merge(eod[["date", "pair", "carry_signal", "mr_signal"]], on=["date", "pair"], how="left")
    out = out.merge(dc[["date", "pair", "momentum_signal"]], on=["date", "pair"], how="left")
    return out.fillna(0)


def build_composite_lookup(sdf):
    lk = {}
    for _, r in sdf.iterrows():
        raw = 0.50*r["fix_signal"] + 0.25*r["carry_signal"] + 0.15*r["momentum_signal"] + 0.10*r["mr_signal"]
        lk[(r["date"], r["pair"])] = float(np.clip(raw * MAX_SKEW_PIPS, -MAX_SKEW_PIPS, MAX_SKEW_PIPS))
    return lk


def alpha_per_rfq(rfqs, lookup):
    ts = pd.to_datetime(rfqs["timestamp"])
    return np.array([lookup.get((ts.iloc[i].date(), rfqs["pair"].iloc[i]), 0.0) for i in range(len(ts))])


# ── Adverse selection simulator ──────────────────────────────────────

def simulate_adverse_selection(n, segments, fix_proximities, notionals, rng):
    """Per-fill adverse selection calibrated to segment + fix proximity + size.

    Returns array of adverse pips (negative = against market maker).
    """
    base_adverse = np.array([ADVERSE_BY_SEGMENT.get(s, -0.8) for s in segments])
    fix_adverse = np.array([ADVERSE_BY_FIX.get(fp, 0.0) for fp in fix_proximities])
    size_adverse = -0.2 * np.log(notionals / 5e6)  # larger → more informed

    mean_adverse = base_adverse + fix_adverse + size_adverse

    # Fat-tail noise: t-distribution with df=4 (heavier than normal)
    noise = rng.standard_t(df=4, size=n) * 0.8
    return mean_adverse + noise


# ── Toxicity filter (logistic regression) ────────────────────────────

def build_toxicity_features(rfqs, adverse_pnl):
    """Feature matrix for toxicity prediction."""
    segment_map = {"hedge_fund": 3, "retail": 2, "real_money": 1, "corporate": 0, "central_bank": 0}
    fix_map = {"at_fix": 3, "pre_fix": 2, "post_fix": 1, "neutral": 0}

    X = pd.DataFrame({
        "segment_code": [segment_map.get(s, 1) for s in rfqs["client_segment"]],
        "fix_code": [fix_map.get(fp, 0) for fp in rfqs["fix_proximity"]],
        "log_notional": np.log(rfqs["notional_usd"].values / 5e6),
        "spread": rfqs["quoted_spread_pips"].values,
    })
    # Label: toxic if adverse selection worse than -1.5p
    y = (adverse_pnl < -1.5).astype(int)
    return X, y


def train_toxicity_filter(X_train, y_train):
    """Train logistic regression to predict toxic flow."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_scaled, y_train)
    return model, scaler


# ── Realistic backtest engine ────────────────────────────────────────

def realistic_backtest(label, optimizer, rfqs, alpha_arr, toxicity_model=None,
                       toxicity_scaler=None, reject_threshold=0.5,
                       internalization_rate=0.0,
                       last_look_ms=0, last_look_reject_bps=0.0,
                       seed=42):
    """
    Internalization: opposite-direction fills net against inventory (no hedge).

    Last-look: after a fill, observe the market move over `last_look_ms`
    milliseconds. If the move is adverse by more than `last_look_reject_bps`
    (in pips), reject the fill. This models the 50-200ms hold window that
    real ECN/single-dealer platforms use. The client's order is returned
    unfilled — the desk avoids the toxic fill entirely.
    """
    rng = np.random.RandomState(seed)
    n = len(rfqs)
    if alpha_arr is None:
        alpha_arr = np.zeros(n)

    rfqs = rfqs.copy()
    rfqs["date"] = pd.to_datetime(rfqs["timestamp"]).dt.date
    pairs = rfqs["pair"].values
    directions = rfqs["direction"].values
    segments = rfqs["client_segment"].values
    fix_proxs = rfqs["fix_proximity"].values
    notionals = rfqs["notional_usd"].values
    timestamps = rfqs["timestamp"].values
    rfq_signs = np.where(directions == "buy_base", 1.0, -1.0)

    wm = optimizer.win_model
    spread_grid = np.linspace(0.05, 2.0, 80)

    daily_shocks = {}
    for d in sorted(rfqs["date"].unique()):
        daily_shocks[d] = {p: rng.randn() * DAILY_VOL.get(p, 60) / np.sqrt(10) for p in G4}

    all_adverse = simulate_adverse_selection(n, segments, fix_proxs, notionals, rng)

    if toxicity_model is not None:
        X_all, _ = build_toxicity_features(rfqs, all_adverse)
        tox_scores = toxicity_model.predict_proba(toxicity_scaler.transform(X_all))[:, 1]
    else:
        tox_scores = np.zeros(n)

    inventory = {p: 0.0 for p in G4}
    lambda_inv = 0.08

    opt_spreads = np.zeros(n)
    was_filled = np.zeros(n, dtype=bool)
    was_rejected = np.zeros(n, dtype=bool)
    was_internalized = np.zeros(n, dtype=bool)
    spread_pnl = np.zeros(n)
    hedge_cost_pnl = np.zeros(n)
    alpha_pnl_arr = np.zeros(n)
    adverse_pnl = np.zeros(n)
    inventory_pnl = np.zeros(n)
    flatten_cost = np.zeros(n)

    n_rejected = 0
    n_internalized = 0
    n_last_look_rejected = 0

    for i in range(n):
        pair = pairs[i]
        hc = L2_HEDGE_COST.get(pair, 0.20)

        if tox_scores[i] > reject_threshold:
            was_rejected[i] = True
            n_rejected += 1
            continue

        tox_widen = max(0, (tox_scores[i] - 0.3) * 2.0)

        ts = timestamps[i]
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        fix_sched = optimizer.fix_alpha.fix_schedule(ts)
        prox = optimizer._alpha_proximity_factor(fix_sched)
        boost = optimizer._regime_boost(fix_sched)
        a_per_trade = alpha_arr[i] * rfq_signs[i] * prox

        inv = inventory[pair]
        fill_dir = rfq_signs[i]
        inv_after = inv + fill_dir
        inv_penalty = lambda_inv * (inv_after ** 2 - inv ** 2)

        # Check if this fill can internalize (opposite direction to inventory)
        can_internalize = (inv * fill_dir < 0)  # opposite signs
        will_internalize = can_internalize and (rng.rand() < internalization_rate)

        # If internalizing: hedge cost = 0 (no market trade needed)
        effective_hc = 0.0 if will_internalize else hc

        probs = 1.0 / (1.0 + np.exp(-wm.steepness * (spread_grid - wm.midpoint)))
        epnls = probs * (spread_grid - effective_hc + a_per_trade - tox_widen) - inv_penalty * probs + boost
        best = int(np.argmax(epnls))
        s = spread_grid[best] + tox_widen

        if fix_sched.get("fix_proximity") == "at_fix":
            s *= fix_sched.get("spread_multiplier", 1.5)

        opt_spreads[i] = s

        fill_prob = 1.0 / (1.0 + np.exp(-wm.steepness * (s - wm.midpoint)))
        filled = rng.rand() < fill_prob
        was_filled[i] = filled

        if filled:
            # ── Last-look: observe market move over hold window ──
            # The adverse_pnl for this fill is a proxy for "what the market
            # did in the next few hundred ms". If it moved against us by
            # more than the threshold, reject the fill.
            if last_look_ms > 0:
                # Noisy last-look: observe true adverse move scaled to the
                # hold window, PLUS microstructure noise that makes the
                # observation imperfect. This is the key realism fix —
                # the desk does NOT see the true adverse; it sees a noisy
                # price tick over 200ms dominated by bid-ask bounce.
                #
                # Signal fraction: ~40% of the 5-min adverse move is
                # visible within 200ms (informed flow front-runs fast).
                ll_fraction = min(last_look_ms / 500.0, 0.6)
                true_signal = all_adverse[i] * ll_fraction

                # Noise: microstructure (bid-ask bounce + random ticks).
                # σ_noise ≈ 0.8 pips for EUR/USD over 200ms, scaled by
                # pair vol. This is LARGER than the signal for most fills,
                # so the rejection is noisy — you reject some good fills
                # and miss some bad ones.
                pair_vol_scale = DAILY_VOL.get(pair, 60) / 55.0
                noise_std = 0.8 * pair_vol_scale * np.sqrt(last_look_ms / 200.0)
                observed = true_signal + rng.randn() * noise_std

                if observed < -last_look_reject_bps:
                    was_filled[i] = False
                    n_last_look_rejected += 1
                    continue

            inventory[pair] += fill_dir

            if will_internalize:
                was_internalized[i] = True
                n_internalized += 1
                spread_pnl[i] = s
                hedge_cost_pnl[i] = 0.0
                adverse_pnl[i] = all_adverse[i] * 0.3
            else:
                spread_pnl[i] = s
                hedge_cost_pnl[i] = -hc
                adverse_pnl[i] = all_adverse[i]

            alpha_pnl_arr[i] = a_per_trade
            daily_shock = daily_shocks[rfqs["date"].iloc[i]].get(pair, 0)
            inventory_pnl[i] = fill_dir * daily_shock * 0.05

        # EOD flatten: only hedge the NET residual
        is_last = True
        if i + 1 < n:
            if rfqs["date"].iloc[i + 1] == rfqs["date"].iloc[i] and pairs[i + 1] == pair:
                is_last = False
        if is_last and abs(inventory[pair]) > 0:
            # Flatten cost from AC model: only on residual net inventory
            from module_c_execution.market_impact import AlmgrenChrissModel
            ac = AlmgrenChrissModel()
            ticker_map = {"EUR/USD": "6E", "GBP/USD": "6B", "JPY/USD": "6J", "AUD/USD": "6A"}
            ticker = ticker_map.get(pair, "6E")
            n_contracts = int(abs(inventory[pair]))
            if n_contracts > 0:
                try:
                    ac_result = ac.cost_for_futures(ticker, n_contracts)
                    fc = float(ac_result.total_cost_pips) if hasattr(ac_result, "total_cost_pips") else float(ac_result["total_cost_pips"])
                except Exception:
                    fc = abs(inventory[pair]) * EOD_FLATTEN_COST.get(pair, 0.40)
            else:
                fc = 0
            ds = daily_shocks[rfqs["date"].iloc[i]].get(pair, 0)
            flatten_cost[i] = -fc + inventory[pair] * ds * 0.15
            inventory[pair] = 0.0

    total_pnl_arr = spread_pnl + hedge_cost_pnl + alpha_pnl_arr + adverse_pnl + inventory_pnl + flatten_cost

    rfqs["total_pnl"] = total_pnl_arr
    rfqs["was_filled"] = was_filled
    rfqs["_spread_pnl"] = spread_pnl
    rfqs["_adverse_pnl"] = adverse_pnl
    rfqs["_inv_pnl"] = inventory_pnl + flatten_cost
    daily = rfqs.groupby("date").agg(
        total=("total_pnl", "sum"),
        n_fills=("was_filled", "sum"),
    )

    active = daily[daily["n_fills"] > 0]
    n_fills = int(was_filled.sum())
    filled_pnl = total_pnl_arr[was_filled]

    if len(active) > 5 and active["total"].std() > 0:
        sharpe = active["total"].mean() / active["total"].std() * np.sqrt(252)
        ds_neg = active["total"][active["total"] < 0]
        sortino = active["total"].mean() / ds_neg.std() * np.sqrt(252) if len(ds_neg) > 1 and ds_neg.std() > 0 else 99
        cum = active["total"].cumsum()
        max_dd = float((cum - cum.cummax()).min())
        win_rate = float((active["total"] > 0).mean())
    else:
        sharpe = sortino = max_dd = win_rate = 0.0

    return {
        "label": label, "n_rfqs": n, "n_fills": n_fills,
        "n_rejected": n_rejected,
        "fill_rate": n_fills / max(n - n_rejected, 1),
        "total_pnl": round(float(total_pnl_arr.sum()), 2),
        "mean_fill_pnl": round(float(filled_pnl.mean()), 4) if n_fills else 0,
        "avg_spread": round(float(opt_spreads[was_filled].mean()), 3) if n_fills else 0,
        "sharpe": round(sharpe, 2),
        "sortino": round(min(sortino, 99), 2),
        "max_dd": round(max_dd, 2),
        "win_rate": round(win_rate, 3),
        "active_days": len(active),
        "daily_df": active,
        "filled_pnl": filled_pnl,
        "n_internalized": n_internalized,
        "n_last_look": n_last_look_rejected,
        "total_spread": round(float(spread_pnl.sum()), 1),
        "total_hedge_cost": round(float(hedge_cost_pnl.sum()), 1),
        "total_adverse": round(float(adverse_pnl.sum()), 1),
        "total_inv": round(float((inventory_pnl + flatten_cost).sum()), 1),
        "total_alpha": round(float(alpha_pnl_arr.sum()), 1),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Pure Basis Composite — Realistic + Toxic Flow Filter")
    print("=" * 80)

    sdf = build_basis_signals()
    composite_lk = build_composite_lookup(sdf)

    gen = FXRFQGenerator()
    rfqs = gen.generate(2000, FX_SPOTS, "2026-01-28", "2026-04-28", seed=42)

    win_model = RealisticWinModel(steepness=-4.0, midpoint=0.6)
    fix_model = FixAlphaModel()
    optimizer = FXQuoteOptimizer(win_model, fix_model, lambda_risk=1e-6)
    optimizer.HEDGE_COST_PIPS = L2_HEDGE_COST

    # ── Phase 1: Train toxicity filter on first half ──
    print("\n[1] Training toxicity filter on first 1000 RFQs...")
    train_rfqs = rfqs.head(1000).copy()
    rng_train = np.random.RandomState(99)
    train_adverse = simulate_adverse_selection(
        len(train_rfqs), train_rfqs["client_segment"].values,
        train_rfqs["fix_proximity"].values, train_rfqs["notional_usd"].values,
        rng_train,
    )
    X_train, y_train = build_toxicity_features(train_rfqs, train_adverse)
    tox_model, tox_scaler = train_toxicity_filter(X_train, y_train)
    train_tox_rate = y_train.mean()
    print(f"    Training toxic rate: {train_tox_rate:.1%}")
    print(f"    Features: {list(X_train.columns)}")
    coefs = dict(zip(X_train.columns, tox_model.coef_[0]))
    print(f"    Coefficients: { {k: round(v, 3) for k, v in coefs.items()} }")
    train_proba = tox_model.predict_proba(tox_scaler.transform(X_train))[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_train, train_proba)
    print(f"    In-sample AUC: {auc:.3f}")

    # ── Phase 2: Backtest on second half ──
    test_rfqs = rfqs.tail(1000).copy().reset_index(drop=True)
    alpha_composite = alpha_per_rfq(test_rfqs, composite_lk)

    print(f"\n[2] Backtesting on held-out 1000 RFQs...")
    print(f"    Adverse selection: segment-calibrated (HF -1.8p, corp -0.3p) + fix (+1.5p at-fix)")
    print(f"    Toxicity threshold: reject if P(toxic) > 0.50")

    results = []

    # A: Baseline — no signal, no filter, nothing
    results.append(realistic_backtest(
        "Baseline (nothing)", optimizer, test_rfqs, None))

    # B: Basis alpha + toxicity filter only
    results.append(realistic_backtest(
        "Basis+filter", optimizer, test_rfqs, alpha_composite,
        toxicity_model=tox_model, toxicity_scaler=tox_scaler, reject_threshold=0.40))

    # C: + 65% internalization
    results.append(realistic_backtest(
        "+65% internalization", optimizer, test_rfqs, alpha_composite,
        toxicity_model=tox_model, toxicity_scaler=tox_scaler, reject_threshold=0.40,
        internalization_rate=0.65))

    # D: + last-look 100ms, reject if adverse > 0.5p
    results.append(realistic_backtest(
        "+LL 100ms/0.5p", optimizer, test_rfqs, alpha_composite,
        toxicity_model=tox_model, toxicity_scaler=tox_scaler, reject_threshold=0.40,
        internalization_rate=0.65, last_look_ms=100, last_look_reject_bps=0.5))

    # E: + last-look 150ms, reject if adverse > 0.3p (tighter)
    results.append(realistic_backtest(
        "+LL 150ms/0.3p", optimizer, test_rfqs, alpha_composite,
        toxicity_model=tox_model, toxicity_scaler=tox_scaler, reject_threshold=0.40,
        internalization_rate=0.65, last_look_ms=150, last_look_reject_bps=0.3))

    # F: + last-look 200ms, reject if adverse > 0.2p (aggressive)
    results.append(realistic_backtest(
        "+LL 200ms/0.2p (full)", optimizer, test_rfqs, alpha_composite,
        toxicity_model=tox_model, toxicity_scaler=tox_scaler, reject_threshold=0.40,
        internalization_rate=0.65, last_look_ms=200, last_look_reject_bps=0.2))

    # Print
    print(f"\n{'=' * 130}")
    hdr = (f"  {'Strategy':<26s} {'Fills':>5s} {'Rej':>4s} {'Itrn':>4s} {'LL':>3s} {'Total':>8s} "
           f"{'Spread':>7s} {'HdgCst':>7s} {'Advers':>7s} {'Inv':>7s} {'Alpha':>6s} "
           f"{'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'Win%':>5s}")
    print(hdr)
    print("  " + "─" * 130)
    for r in results:
        print(f"  {r['label']:<26s} {r['n_fills']:>5d} {r['n_rejected']:>4d} "
              f"{r.get('n_internalized',0):>4d} {r.get('n_last_look',0):>3d} "
              f"{r['total_pnl']:>+8.1f} "
              f"{r['total_spread']:>+7.1f} {r.get('total_hedge_cost',0):>+7.1f} "
              f"{r['total_adverse']:>+7.1f} {r['total_inv']:>+7.1f} "
              f"{r['total_alpha']:>+6.1f} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1f} {r['win_rate']:>5.1%}")

    best = results[-1]  # full system
    worst = results[0]  # no signal, no filter
    print(f"\n  Full system vs baseline:")
    print(f"    Sharpe:  {worst['sharpe']:.2f} → {best['sharpe']:.2f}")
    print(f"    PnL:     {worst['total_pnl']:+.1f} → {best['total_pnl']:+.1f}  ({best['total_pnl'] - worst['total_pnl']:+.1f}p)")
    print(f"    Rejected: {best['n_rejected']} toxic RFQs avoided")

    # Plot
    set_jpm_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Equity curves
    ax = axes[0, 0]
    colors_map = [JPM_COLORS["gray"], JPM_COLORS["gold"], JPM_COLORS["red"], JPM_COLORS["blue"], JPM_COLORS["green"]]
    for r, c in zip(results, colors_map):
        cum = r["daily_df"]["total"].cumsum()
        ax.plot(cum.index, cum.values, color=c, linewidth=1.5, label=r["label"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=6.5, loc="lower left"); ax.set_ylabel("Cumulative P&L (pips)")
    ax.set_title("Equity Curves", fontweight="bold"); ax.grid(True, alpha=0.3)

    # P&L decomposition: full system
    ax = axes[0, 1]
    comps = ["total_spread", "total_adverse", "total_inv", "total_alpha"]
    comp_labels = ["Spread", "Adverse\nSel.", "Inv+Flat", "Alpha"]
    comp_colors = [JPM_COLORS["blue"], JPM_COLORS["red"], JPM_COLORS["gold"], JPM_COLORS["green"]]
    for j, r in enumerate([results[0], results[-1]]):
        vals = [r[c] for c in comps]
        x = np.arange(len(vals)) + (j - 0.5) * 0.35
        label = "No signal" if j == 0 else "Full system"
        ax.bar(x, vals, width=0.3, color=comp_colors, alpha=0.5 + 0.3 * j, edgecolor="black" if j == 1 else "none",
               label=label)
    ax.set_xticks(range(len(comp_labels))); ax.set_xticklabels(comp_labels)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("P&L (pips)"); ax.set_title("P&L Decomposition", fontweight="bold")

    # Fill P&L distribution
    ax = axes[1, 0]
    for r, c, lbl in [(results[0], JPM_COLORS["gray"], "No filter"),
                       (results[-1], JPM_COLORS["blue"], "Full system")]:
        if len(r["filled_pnl"]) > 0:
            ax.hist(r["filled_pnl"], bins=40, alpha=0.5, color=c,
                    label=f"{lbl} μ={r['mean_fill_pnl']:+.3f}p")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Per-Fill P&L (pips)"); ax.set_title("Fill P&L Distribution", fontweight="bold")
    ax.legend(fontsize=8)

    # Sharpe bar chart
    ax = axes[1, 1]
    labels = [r["label"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bars = ax.bar(range(len(results)), sharpes, color=colors_map[:len(results)], alpha=0.8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Sharpe (annualized)"); ax.set_title("After-Cost Sharpe", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, s in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, s + (0.1 if s >= 0 else -0.3),
                f"{s:.2f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Pure Basis Composite + Toxic Flow Filter\n"
                 "(logistic regression on segment/fix/size → reject P(toxic) > threshold)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PROJECT_ROOT / "output" / "backtest_basis_composite.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out}")


if __name__ == "__main__":
    main()
