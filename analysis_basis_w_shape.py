"""Intraday calendar-spread basis W-shape analysis across G4 pairs.

Uses true intraday calendar spread: log(F_back / F_front) × 1e4, computed
from CME OHLCV-1m data for front (.c.0) and back (.c.1) continuous contracts.

Spot moves cancel (both contracts track the same underlying). What remains
is the implied forward-forward funding rate spread between two quarterly
expiries — a pure basis signal. Intraday changes in this spread reflect
shifts in USD-funding pressure within the trading session.

Metric:
    basis_change_bps = Δ log(F_back/F_front) × 1e4, per 15-min bucket
    cum_basis_bps    = cumulative sum within each session

Stratified by:
    calm         = >=2 biz days from month-end
    month_end    = last 2 biz days of any month
    quarter_end  = last biz day of Mar/Jun/Sep/Dec
    year_end     = last biz day of Dec

Output: output/w_shape_basis_g4.png (2×4 small multiples, full-day coverage
        with Tokyo, ECB, and London fix lines)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.plot_style import JPM_COLORS, set_jpm_style

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE = PROJECT_ROOT / "data" / "cme_15m_basis_cache.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output"
G4 = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]

FIX_TIMES = [("Tokyo 00:55", 3), ("ECB 13:15", 53), ("London 16:00", 64)]


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


def compute_cum_basis_by_bucket(df, session_types):
    """Cumulative basis_change_bps within each session, averaged by day-type and bucket."""
    df = df.dropna(subset=["basis_change_bps"]).copy()
    df["day_type"] = df["date"].map(session_types)

    # Cumulative within each (pair, date) session
    df = df.sort_values(["pair", "date", "bucket"])
    df["cum_basis"] = df.groupby(["pair", "date"])["basis_change_bps"].cumsum()

    # Average across dates in each day-type, per (pair, bucket)
    grp = df.groupby(["day_type", "pair", "bucket"])["cum_basis"]
    agg = grp.agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]
    return agg


def compute_fix_windows(df, session_types):
    """Mean basis_change_bps within ±60min fix windows, by day-type."""
    df = df.dropna(subset=["basis_change_bps"]).copy()
    df["day_type"] = df["date"].map(session_types)
    windows = [(-60, -30), (-30, -15), (-15, 0), (0, 15), (15, 30), (30, 60)]
    rows = []
    for fix_name, fix_bucket in FIX_TIMES:
        df["min_from_fix"] = (df["bucket"] - fix_bucket) * 15
        for start, end in windows:
            m = (df["min_from_fix"] >= start) & (df["min_from_fix"] < end)
            for (pair, dt), g in df[m].groupby(["pair", "day_type"]):
                v = g["basis_change_bps"]
                n = len(v)
                rows.append({
                    "fix": fix_name, "pair": pair, "day_type": dt,
                    "window": f"[{start:+d},{end:+d})",
                    "window_start": start,
                    "mean_bps": v.mean() if n else np.nan,
                    "t_stat": v.mean() / (v.std() / np.sqrt(n)) if n > 1 and v.std() > 0 else np.nan,
                    "n": n,
                })
    return pd.DataFrame(rows)


def plot(bucket_agg, fix_agg, session_counts, out_path):
    set_jpm_style()
    fig, axes = plt.subplots(2, len(G4), figsize=(4.5 * len(G4), 9), sharey="row")

    COLORS = {
        "calm": JPM_COLORS["gray"],
        "month_end": JPM_COLORS["gold"],
        "quarter_end": JPM_COLORS["blue"],
        "year_end": JPM_COLORS["red"],
    }
    DAY_ORDER = ["calm", "month_end", "quarter_end", "year_end"]
    labels = {
        "calm": f"calm (n={session_counts.get('calm', 0)})",
        "month_end": f"month-end (n={session_counts.get('month_end', 0)})",
        "quarter_end": f"quarter-end (n={session_counts.get('quarter_end', 0)})",
        "year_end": f"year-end (n={session_counts.get('year_end', 0)})",
    }

    # Top row: cumulative basis through the day
    all_buckets = sorted(bucket_agg["bucket"].unique())
    tick_buckets = list(range(0, 96, 8))
    tick_labels = [f"{(b*15)//60:02d}:{(b*15)%60:02d}" for b in tick_buckets]

    for col, pair in enumerate(G4):
        ax = axes[0, col]
        for dt in DAY_ORDER:
            sub = bucket_agg[(bucket_agg["pair"] == pair) & (bucket_agg["day_type"] == dt)]
            if sub.empty:
                continue
            sub = sub.sort_values("bucket")
            ax.plot(sub["bucket"], sub["mean"], color=COLORS[dt], linewidth=1.8,
                    label=labels[dt], alpha=0.9)
            if dt == "calm" and sub["count"].max() > 2:
                ax.fill_between(sub["bucket"], sub["mean"] - sub["ci95"],
                                sub["mean"] + sub["ci95"], color=COLORS[dt], alpha=0.12)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        for fix_name, fix_b in FIX_TIMES:
            ax.axvline(fix_b, color=JPM_COLORS["red"], linestyle="--",
                       linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.2)
        ax.set_title(pair, fontsize=11, fontweight="bold")
        ax.set_xticks(tick_buckets)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        if col == 0:
            ax.set_ylabel("Cumulative basis change (bps)\nlog(F_back/F_front)")
            ax.legend(loc="best", fontsize=6.5, framealpha=0.9)

    # Bottom row: London fix-window bars (side-by-side)
    london_fix = fix_agg[fix_agg["fix"] == "London 16:00"]
    windows = [(-60, -30), (-30, -15), (-15, 0), (0, 15), (15, 30), (30, 60)]
    win_labels = [f"[{s:+d},{e:+d})" for s, e in windows]
    n_wins = len(win_labels)
    present_types = [dt for dt in DAY_ORDER if dt in london_fix["day_type"].values]
    n_types = len(present_types)
    bar_w = 0.8 / max(n_types, 1)

    win_starts = [s for s, _ in windows]
    for col, pair in enumerate(G4):
        ax = axes[1, col]
        for i, dt in enumerate(present_types):
            sub = london_fix[(london_fix["pair"] == pair) & (london_fix["day_type"] == dt)]
            if sub.empty:
                continue
            full = pd.DataFrame({"window_start": win_starts})
            merged = full.merge(sub[["window_start", "mean_bps"]], on="window_start", how="left")
            x = np.arange(n_wins) + (i - n_types / 2 + 0.5) * bar_w
            ax.bar(x, merged["mean_bps"].fillna(0), width=bar_w, color=COLORS[dt],
                   alpha=0.85, label=labels[dt] if col == 0 else None)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_xticks(np.arange(n_wins))
        ax.set_xticklabels(win_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Window around London fix (min)")
        if col == 0:
            ax.set_ylabel("Mean basis change (bps)")
            ax.legend(loc="best", fontsize=6.5, framealpha=0.9)

    fig.suptitle(
        "Intraday Calendar-Spread Basis Pattern Across G4 Pairs\n"
        r"$\Delta\log(F_{back}/F_{front}) \times 10^4$  —  "
        "pure funding-pressure signal (spot cancels)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    nc = session_counts
    fig.text(
        0.5, -0.01,
        f"Sample: Dec 2024 – Apr 2026, CME OHLCV-1m, 15-min aggregation.  "
        f"Sessions: {nc.get('calm',0)} calm, {nc.get('month_end',0)} month-end, "
        f"{nc.get('quarter_end',0)} quarter-end, {nc.get('year_end',0)} year-end.  "
        f"Dashed red lines: Tokyo (00:55), ECB (13:15), London (16:00) fixes.",
        ha="center", fontsize=7.5, style="italic",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    dates = sorted(df["date"].unique())
    stypes = label_sessions(dates)
    counts = pd.Series(stypes).value_counts().to_dict()
    print(f"  Loaded {len(df):,} rows, {len(dates)} sessions")
    print(f"  Session counts: {counts}")

    bucket_agg = compute_cum_basis_by_bucket(df, stypes)
    fix_agg = compute_fix_windows(df, stypes)

    # Print London fix-window table
    london = fix_agg[fix_agg["fix"] == "London 16:00"].copy()
    for dt in ["calm", "month_end", "quarter_end", "year_end"]:
        sub = london[london["day_type"] == dt]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="pair", columns="window", values="mean_bps")
        print(f"\n  -- {dt} (London fix windows, mean basis change bps) --")
        print(wide.round(2).to_string())

    out = OUTPUT_DIR / "w_shape_basis_g4.png"
    plot(bucket_agg, fix_agg, counts, out)
    print(f"\n  Plot saved: {out}")


if __name__ == "__main__":
    main()
