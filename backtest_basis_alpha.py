"""Backtest three alpha strategies on intraday calendar-spread basis.

Strategies:
    1. Basis Fix-Reversal (V-shape trade on stress days)
    2. Basis Momentum (trend-following on daily basis changes)
    3. Basis Mean-Reversion (z-score fade)

Uses cme_15m_basis_cache.parquet (Dec 2024 – Apr 2026, 435 sessions).
Hedge costs derived from Almgren-Chriss calibrated to CME MBP-10 L2 book.

Output: printed metrics table + output/backtest_basis_alpha.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.plot_style import JPM_COLORS, set_jpm_style

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE = PROJECT_ROOT / "data" / "cme_15m_basis_cache.parquet"
OUTPUT = PROJECT_ROOT / "output" / "backtest_basis_alpha.png"
G4 = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]

LONDON_FIX_BUCKET = 64  # 16:00 UTC

# Round-trip hedge cost per calendar-spread trade (both legs, entry+exit) in
# basis-bps. Derived from Almgren-Chriss realised TWAP slippage on CME MBP-10:
#   cost_bps ≈ 2 × (half_spread + AC_slippage) / F_mid × 10000 × 2 (entry+exit)
# Conservative estimates including bid-ask + impact on both legs:
HEDGE_COST_RT_BPS = {
    "EUR/USD": 3.0,
    "GBP/USD": 4.5,
    "JPY/USD": 5.0,
    "AUD/USD": 4.0,
}

BUCKETS_PER_DAY = 96  # 24h × 4 per hour


def load() -> pd.DataFrame:
    df = pd.read_parquet(CACHE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["bucket"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    return df


def label_sessions(dates):
    out = {}
    for d in dates:
        d_ts = pd.Timestamp(d)
        n1 = d_ts + pd.tseries.offsets.BDay(1)
        n2 = d_ts + pd.tseries.offsets.BDay(2)
        is_last_bd = n1.month != d_ts.month
        is_penult_bd = n2.month != d_ts.month and n1.month == d_ts.month
        if is_last_bd and d_ts.month == 12:
            out[d] = "year_end"
        elif is_last_bd and d_ts.month in (3, 6, 9):
            out[d] = "quarter_end"
        elif is_last_bd or is_penult_bd:
            out[d] = "month_end"
        else:
            out[d] = "calm"
    return out


# ── Strategy 1: Basis Fix-Reversal (V-shape on stress days) ──────────

def strategy_fix_reversal(df: pd.DataFrame, session_types: dict) -> pd.DataFrame:
    """Trade the V-shaped basis reversal at the London fix on stress days.

    On month-end/quarter-end/year-end days:
      - Enter at [-30, -15) before London fix: position = -1 (short basis,
        riding the pre-fix widening)
      - Hold through fix reversal, exit at [+30, +45) after fix
    On calm days: flat.
    """
    df = df.copy()
    df["day_type"] = df["date"].map(session_types)
    df["position"] = 0.0

    stress = df["day_type"].isin(["month_end", "quarter_end", "year_end"])
    pre_fix = (df["bucket"] >= LONDON_FIX_BUCKET - 2) & (df["bucket"] < LONDON_FIX_BUCKET)
    at_fix = (df["bucket"] >= LONDON_FIX_BUCKET) & (df["bucket"] < LONDON_FIX_BUCKET + 1)
    post_fix = (df["bucket"] >= LONDON_FIX_BUCKET + 1) & (df["bucket"] < LONDON_FIX_BUCKET + 3)

    # Pre-fix: short basis (basis tends to widen → then reverses at fix)
    df.loc[stress & pre_fix, "position"] = -1.0
    # At fix: flip to long (catch the reversal)
    df.loc[stress & at_fix, "position"] = 1.0
    # Post-fix: hold long (reversal continues)
    df.loc[stress & post_fix, "position"] = 1.0

    return df[["timestamp", "date", "pair", "bucket", "basis_change_bps", "position"]]


# ── Strategy 2: Basis Momentum ───────────────────────────────────────

def strategy_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Daily momentum: if 5-day trailing basis has been widening, go long.

    Signal = sign of 5-day EMA of daily basis change.
    Position is held all day, rebalanced at session open.
    """
    daily = (
        df.groupby(["date", "pair"])["basis_change_bps"]
          .sum()
          .reset_index()
          .rename(columns={"basis_change_bps": "daily_basis_chg"})
    )
    daily = daily.sort_values(["pair", "date"])
    daily["ema5"] = daily.groupby("pair")["daily_basis_chg"].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean()
    )
    daily["signal"] = np.sign(daily["ema5"])

    df = df.merge(daily[["date", "pair", "signal"]], on=["date", "pair"], how="left")
    df["signal"] = df["signal"].fillna(0)
    df["position"] = df["signal"]
    return df[["timestamp", "date", "pair", "bucket", "basis_change_bps", "position"]]


# ── Strategy 3: Basis Mean-Reversion ─────────────────────────────────

def strategy_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """Fade extreme basis levels using 20-day z-score.

    z > 1.5 → short basis (expect reversion down)
    z < -1.5 → long basis (expect reversion up)
    Otherwise flat.
    """
    daily = (
        df.groupby(["date", "pair"])["log_basis_bps"]
          .last()
          .reset_index()
          .rename(columns={"log_basis_bps": "eod_basis"})
    )
    daily = daily.sort_values(["pair", "date"])
    daily["ma20"] = daily.groupby("pair")["eod_basis"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    daily["std20"] = daily.groupby("pair")["eod_basis"].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    daily["z_score"] = (daily["eod_basis"] - daily["ma20"]) / daily["std20"].clip(lower=0.01)
    daily["signal"] = 0.0
    daily.loc[daily["z_score"] > 1.5, "signal"] = -1.0
    daily.loc[daily["z_score"] < -1.5, "signal"] = 1.0

    df = df.merge(daily[["date", "pair", "signal"]], on=["date", "pair"], how="left")
    df["signal"] = df["signal"].fillna(0)
    df["position"] = df["signal"]
    return df[["timestamp", "date", "pair", "bucket", "basis_change_bps", "position"]]


# ── Backtester ────────────────────────────────────────────────────────

def add_log_basis(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has log_basis_bps for mean-reversion strategy."""
    if "log_basis_bps" not in df_raw.columns:
        df_raw["log_basis_bps"] = 0.0
    return df_raw


def backtest(positions: pd.DataFrame, label: str) -> dict:
    """Compute P&L with hedge costs.

    P&L per bucket = position(t-1) × basis_change_bps(t)
    Costs charged on every position change (|Δposition| × half-RT-cost).
    """
    df = positions.copy()
    df = df.sort_values(["pair", "timestamp"]).reset_index(drop=True)

    # Lagged position for P&L: earn on position held entering the bucket
    df["pos_prev"] = df.groupby("pair")["position"].shift(1).fillna(0)
    df["gross_pnl_bps"] = df["pos_prev"] * df["basis_change_bps"]

    # Transaction costs: charged on |Δposition|
    df["delta_pos"] = (df["position"] - df["pos_prev"]).abs()
    df["cost_bps"] = df["pair"].map(HEDGE_COST_RT_BPS).fillna(4.0) * df["delta_pos"] / 2.0
    df["net_pnl_bps"] = df["gross_pnl_bps"] - df["cost_bps"]

    # Aggregate per pair
    pair_stats = []
    for pair in G4:
        p = df[df["pair"] == pair]
        if p.empty or p["pos_prev"].abs().sum() == 0:
            continue
        gross = p["gross_pnl_bps"].sum()
        cost = p["cost_bps"].sum()
        net = p["net_pnl_bps"].sum()
        n_trades = (p["delta_pos"] > 0).sum()
        # Daily P&L for Sharpe
        p_daily = p.groupby("date")["net_pnl_bps"].sum()
        p_daily = p_daily[p_daily != 0]  # only days with positions
        if len(p_daily) < 5:
            continue
        sharpe = p_daily.mean() / p_daily.std() * np.sqrt(252) if p_daily.std() > 0 else 0
        sortino_denom = p_daily[p_daily < 0].std()
        sortino = p_daily.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else np.inf
        cum = p_daily.cumsum()
        max_dd = (cum - cum.cummax()).min()
        calmar = (p_daily.mean() * 252) / abs(max_dd) if max_dd < 0 else np.inf
        win_rate = (p_daily > 0).mean()
        pair_stats.append({
            "pair": pair,
            "gross_bps": round(gross, 1),
            "cost_bps": round(cost, 1),
            "net_bps": round(net, 1),
            "n_trades": n_trades,
            "n_active_days": len(p_daily),
            "daily_mean_bps": round(p_daily.mean(), 3),
            "daily_std_bps": round(p_daily.std(), 3),
            "sharpe": round(sharpe, 2),
            "sortino": round(min(sortino, 99), 2),
            "max_dd_bps": round(max_dd, 1),
            "calmar": round(min(calmar, 99), 2),
            "win_rate": round(win_rate, 3),
        })

    # Portfolio (equal-weight across pairs)
    port_daily = df.groupby("date")["net_pnl_bps"].sum()
    port_daily = port_daily[port_daily != 0]
    if len(port_daily) >= 5:
        sharpe_p = port_daily.mean() / port_daily.std() * np.sqrt(252) if port_daily.std() > 0 else 0
        sort_d = port_daily[port_daily < 0].std()
        sortino_p = port_daily.mean() / sort_d * np.sqrt(252) if sort_d > 0 else np.inf
        cum_p = port_daily.cumsum()
        dd_p = (cum_p - cum_p.cummax()).min()
        calmar_p = (port_daily.mean() * 252) / abs(dd_p) if dd_p < 0 else np.inf
        pair_stats.append({
            "pair": "PORTFOLIO",
            "gross_bps": round(df["gross_pnl_bps"].sum(), 1),
            "cost_bps": round(df["cost_bps"].sum(), 1),
            "net_bps": round(df["net_pnl_bps"].sum(), 1),
            "n_trades": int((df["delta_pos"] > 0).sum()),
            "n_active_days": len(port_daily),
            "daily_mean_bps": round(port_daily.mean(), 3),
            "daily_std_bps": round(port_daily.std(), 3),
            "sharpe": round(sharpe_p, 2),
            "sortino": round(min(sortino_p, 99), 2),
            "max_dd_bps": round(dd_p, 1),
            "calmar": round(min(calmar_p, 99), 2),
            "win_rate": round((port_daily > 0).mean(), 3),
        })

    return {"label": label, "stats": pd.DataFrame(pair_stats), "daily_pnl": port_daily}


# ── Plotting ──────────────────────────────────────────────────────────

def plot_results(results: list[dict], out_path: Path):
    set_jpm_style()
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), gridspec_kw={"width_ratios": [2, 1]})
    if n == 1:
        axes = axes.reshape(1, -1)

    colors = [JPM_COLORS["blue"], JPM_COLORS["gold"], JPM_COLORS["red"]]
    for i, res in enumerate(results):
        cum = res["daily_pnl"].cumsum()
        ax_cum = axes[i, 0]
        ax_cum.plot(cum.index, cum.values, color=colors[i], linewidth=1.5)
        ax_cum.fill_between(cum.index, 0, cum.values, alpha=0.15, color=colors[i])
        ax_cum.axhline(0, color="black", linewidth=0.5)
        ax_cum.set_title(f"{res['label']}  (Sharpe: {res['stats'][res['stats']['pair']=='PORTFOLIO']['sharpe'].values[0]:.2f})",
                         fontsize=11, fontweight="bold")
        ax_cum.set_ylabel("Cumulative P&L (bps)")
        ax_cum.grid(True, alpha=0.3)

        # Metrics table
        ax_tbl = axes[i, 1]
        ax_tbl.axis("off")
        port = res["stats"][res["stats"]["pair"] == "PORTFOLIO"]
        if not port.empty:
            r = port.iloc[0]
            text = (
                f"Gross P&L:     {r['gross_bps']:+.1f} bps\n"
                f"Hedge Cost:    {r['cost_bps']:.1f} bps\n"
                f"Net P&L:       {r['net_bps']:+.1f} bps\n"
                f"─────────────────────\n"
                f"Sharpe (ann):  {r['sharpe']:.2f}\n"
                f"Sortino:       {r['sortino']:.2f}\n"
                f"Max Drawdown:  {r['max_dd_bps']:.1f} bps\n"
                f"Calmar:        {r['calmar']:.2f}\n"
                f"Win Rate:      {r['win_rate']:.1%}\n"
                f"─────────────────────\n"
                f"Active Days:   {r['n_active_days']}\n"
                f"Trades:        {r['n_trades']}\n"
                f"Daily Mean:    {r['daily_mean_bps']:+.3f} bps\n"
                f"Daily Std:     {r['daily_std_bps']:.3f} bps"
            )
            ax_tbl.text(0.05, 0.95, text, transform=ax_tbl.transAxes,
                        fontsize=9, fontfamily="monospace", va="top")

    fig.suptitle("Cross-Currency Basis Alpha Strategies — After-Cost Backtest\n"
                 "(Dec 2024 – Apr 2026, 435 sessions, 15-min CME calendar-spread basis)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 80)
    print("Cross-Currency Basis Alpha Backtest")
    print("=" * 80)

    df_raw = load()
    df_raw = add_log_basis(df_raw)
    dates = sorted(df_raw["date"].unique())
    stypes = label_sessions(dates)
    type_counts = pd.Series(stypes).value_counts()
    print(f"\n  Data: {len(df_raw):,} rows, {len(dates)} sessions")
    print(f"  Sessions: {type_counts.to_dict()}")
    print(f"  Hedge costs (RT bps): {HEDGE_COST_RT_BPS}")
    print()

    # Run strategies
    pos1 = strategy_fix_reversal(df_raw, stypes)
    pos2 = strategy_momentum(df_raw)
    pos3 = strategy_mean_reversion(df_raw)

    results = []
    for label, pos_df in [
        ("1. Basis Fix-Reversal (V-shape on stress days)", pos1),
        ("2. Basis Momentum (5d EMA trend-follow)", pos2),
        ("3. Basis Mean-Reversion (20d z-score fade)", pos3),
    ]:
        res = backtest(pos_df, label)
        results.append(res)
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")
        print(res["stats"].to_string(index=False))

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("  PORTFOLIO COMPARISON (after costs)")
    print(f"{'=' * 80}")
    rows = []
    for r in results:
        port = r["stats"][r["stats"]["pair"] == "PORTFOLIO"]
        if not port.empty:
            p = port.iloc[0]
            rows.append({
                "Strategy": r["label"].split("(")[0].strip(),
                "Net P&L (bps)": f"{p['net_bps']:+.1f}",
                "Sharpe": f"{p['sharpe']:.2f}",
                "Sortino": f"{p['sortino']:.2f}",
                "Max DD (bps)": f"{p['max_dd_bps']:.1f}",
                "Win Rate": f"{p['win_rate']:.1%}",
                "Active Days": int(p["n_active_days"]),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plot_results(results, OUTPUT)
    print(f"\n  Plot saved: {OUTPUT}")


if __name__ == "__main__":
    main()
