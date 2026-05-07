"""
Full Demo Script: FX Fix-Aware Market Making (W-Shaped Pattern)
Run with:  python run_full_demo.py

Implements Krohn, Mueller & Whelan (Journal of Finance, 2024):
USD systematically appreciates before each of the three daily FX fixes
(Tokyo 00:55 UTC, ECB 13:15 UTC, London WMR 16:00 UTC) and depreciates after.

Pipeline:
  Step 0 - Infrastructure (KDB+ startup, Databento CME FX futures tick data)
  Step 1 - FX Curve Construction (USD OIS, EUR/GBP OIS proxies, FX forwards via CIP)
  Step 2 - FX Pricing & Risk (portfolio NPV, spot delta, cross-gamma, VaR, scenarios)
  Step 3 - Fix Alpha Signals & Quoting (W-shape, carry, momentum, MR, optimizer)
  Step 4 - Hedge Sizing & Market Impact (delta hedge -> CME FX futures, Almgren-Chriss)
  Step 5 - Execution (TWAP/VWAP/Adaptive, L2 slippage backtest via KDB+)
  Step 6 - P&L Analysis (fix-conditioned markouts, decomposition)
  Step 7 - C++ Kernel Benchmark
"""

import sys
import time
import subprocess
import warnings
from pathlib import Path
from datetime import datetime, timedelta, date

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

import os
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5"

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import logging
from shared.plot_style import set_jpm_style, JPM_COLORS
from config import setup_logging, get_fred_api_key, KDB_HOST, KDB_PORT
set_jpm_style()
setup_logging()

# Silence library loggers -- demo output is via print(), only show errors
logging.getLogger().setLevel(logging.ERROR)

output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# Tick data: use date with REAL Databento MBP-10 data (London session around WMR fix)
# Primary cache: data/databento_cache/mbp10_fx_2026-03-18.parquet (real CME tick data)
TICK_DATA_DATE = "2026-03-18"
TICK_START_TIME = "13:00"
TICK_END_TIME = "17:00"
# Valuation date: use yesterday (FRED publishes Treasury yields with 1-day lag)
VALUATION_DATE = "2026-04-28"
FX_TICKERS = ["6E", "6B", "6J", "6A"]
G4_PAIRS = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]


###############################################################################
# STEP 0: Infrastructure (KDB+ & Tick Data)
###############################################################################

print("\n" + "=" * 70)
print("  STEP 0: Infrastructure")
print("  (KDB+ Startup, Databento CME FX Futures Tick Data -> KDB+)")
print("=" * 70)

from shared.kdb_interface import KDBInterface
from shared.databento_loader import fetch_ticker, CACHE_DIR

# --- 0a: Start KDB+ if not already running ---
print("\n-- 0a: KDB+ Server --")
kdb_proc = None
kdb = None

try:
    kdb = KDBInterface(host=KDB_HOST, port=KDB_PORT)
    print(f"  KDB+ already running at {KDB_HOST}:{KDB_PORT}")
    kdb.close()
    kdb = None
except Exception:
    # Start KDB+ as background process
    q_candidates = []
    qhome_env = os.environ.get("QHOME", "")
    if qhome_env:
        q_candidates.append(Path(qhome_env) / "q.exe")
        q_candidates.append(Path(qhome_env) / "w64" / "q.exe")
    q_candidates.append(Path("C:/Users/amand/Downloads/w64/w64/q.exe"))

    q_exe = None
    for p in q_candidates:
        if p.exists():
            q_exe = p
            break

    if q_exe is None:
        print("  KDB+ not found -- using in-memory fallback (pandas DataFrames)")
        print("  Set QHOME environment variable to enable KDB+")
    else:
        qhome = None
        for candidate in [q_exe.parent, q_exe.parent.parent]:
            if (candidate / "q.k").exists():
                qhome = str(candidate)
                break
        if qhome is None:
            qhome = str(q_exe.parent)

        print(f"  Starting KDB+ from {q_exe} (QHOME={qhome}) on port {KDB_PORT}...")
        env = os.environ.copy()
        env["QHOME"] = qhome
        kdb_proc = subprocess.Popen(
            [str(q_exe), "-p", str(KDB_PORT)],
            env=env,
            cwd=str(q_exe.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for attempt in range(20):
            time.sleep(0.5)
            try:
                kdb = KDBInterface(host=KDB_HOST, port=KDB_PORT)
                kdb.close()
                kdb = None
                break
            except Exception:
                pass
        else:
            print("  ERROR: KDB+ failed to start within 10 seconds")
            if kdb_proc:
                kdb_proc.terminate()
            kdb_proc = None

        if kdb_proc:
            print(f"  KDB+ started (PID {kdb_proc.pid})")

# --- 0b: Create tables ---
print("\n-- 0b: Create KDB+ Tables --")
kdb = KDBInterface(host=KDB_HOST, port=KDB_PORT)
kdb.create_tables()
counts = kdb.table_counts()
print(f"  Tables created: {counts}")

# --- 0c: Load tick data from Databento (real cached parquet; synthetic only if cache missing) ---
print(f"\n-- 0c: Load MBP-10 Tick Data ({TICK_DATA_DATE}) --")
tick_count = kdb.tick_data_count()
if tick_count > 0:
    print(f"  KDB+ already has {tick_count} tick_data rows -- skipping reload")
else:
    sampled_frames = []
    total_raw = 0
    for t in FX_TICKERS:
        t_df = fetch_ticker(t, TICK_DATA_DATE, start_time=TICK_START_TIME, end_time=TICK_END_TIME)
        total_raw += len(t_df)
        if len(t_df) > 1000:
            step = len(t_df) // 1000
            t_df = t_df.iloc[::step].head(1000)
        sampled_frames.append(t_df)
    tick_sampled = pd.concat(sampled_frames, ignore_index=True)
    # Fill NaN prices/sizes with 0
    for i in range(10):
        for prefix in ["bid_px_", "ask_px_"]:
            tick_sampled[f"{prefix}{i}"] = tick_sampled[f"{prefix}{i}"].fillna(0.0)
        for prefix in ["bid_sz_", "ask_sz_"]:
            tick_sampled[f"{prefix}{i}"] = tick_sampled[f"{prefix}{i}"].fillna(0).astype(int)
    print(f"  Loaded {total_raw} total tick rows from Databento cache")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Sampled {len(tick_sampled)} snapshots for KDB+ (from {total_raw} total)")

    # Push into KDB+ in small batches
    print("  Inserting into KDB+ tick_data table...")
    n = len(tick_sampled)
    batch_size = 100
    for start in range(0, n, batch_size):
        batch = tick_sampled.iloc[start:start + batch_size]
        kdb.write_tick_data_bulk(batch)
    tick_count = kdb.tick_data_count()
    print(f"  KDB+ tick_data: {tick_count} rows loaded")

# Show per-ticker counts
for t in FX_TICKERS:
    tc = kdb.tick_data_count(t)
    print(f"    {t}: {tc} snapshots")

kdb.close()


###############################################################################
# STEP 1: FX Curve Construction
###############################################################################

print("\n" + "=" * 70)
print("  STEP 1: FX Curve Construction")
print("  (USD OIS, EUR/GBP OIS Proxies, FX Forwards via CIP)")
print("=" * 70)

from module_a_curves.data_loader import FXDataLoader
from module_a_curves.curve_bootstrapper import CurveBootstrapper, CurveInstrument
from module_a_curves.fx_forward_curve import FXForwardCurve, CrossCurrencyBasis

loader = FXDataLoader(fred_api_key=get_fred_api_key())

# --- 1a: FX spot rates ---
print("\n-- 1a: FX Spot Rates --")
fx_spots = loader.get_fx_spot_rates(VALUATION_DATE)
print(f"  FX Spot Rates (as of {VALUATION_DATE}):")
for pair, rate in fx_spots.items():
    if pair in G4_PAIRS:
        print(f"    {pair}: {rate:.4f}")

# --- 1b: USD OIS Curve (SOFR SR3 futures strip via Databento) ---
print("\n-- 1b: USD OIS Curve (SOFR SR3 strip via Databento) --")

from shared.databento_curve_loader import fetch_sofr_strip
from module_a_curves.sofr_futures_bootstrap import bootstrap_usd_ois_from_strip

sr3_strip = fetch_sofr_strip(VALUATION_DATE)
if sr3_strip is None or sr3_strip.empty:
    raise RuntimeError(
        f"SOFR SR3 strip cache is empty for {VALUATION_DATE}. "
        f"Run `python prefetch_data.py --yes` to populate the cache."
    )

# Real overnight SOFR fixing from FRED to anchor the front of the curve.
# Falls back to SR3.c.0 implied rate if the FRED fixing is unavailable.
on_rate_real = None
try:
    _ois = loader.get_ois_rates(VALUATION_DATE)
    if _ois.get("SOFR") and _ois["SOFR"] > 0:
        on_rate_real = float(_ois["SOFR"])
except Exception:
    pass
on_rate_used = on_rate_real if on_rate_real is not None else float(sr3_strip.iloc[0]["implied_rate"])
on_rate_source = "FRED SOFR fixing" if on_rate_real is not None else "SR3.c.0 implied (proxy fallback)"

t0 = time.perf_counter()
usd_curve = bootstrap_usd_ois_from_strip(
    sr3_strip, valuation_date=VALUATION_DATE, overnight_rate=on_rate_real,
)
t_curve = time.perf_counter() - t0

# Build usd_instruments from the strip so downstream FXRiskAnalytics still works.
#
# Each SR3 contract settles to the average daily SOFR over a 3-month period
# starting at its IMM date. We represent each contract as a zero-rate
# deposit anchored at the reference-period END (expiry_years + 0.25) with
# the contract's implied rate. This is the discount-factor-equivalent of
# the futures-strip bootstrap (no "swap" abuse), and it is what downstream
# tools that only understand deposits/swaps actually consume.
#
# Convention: 1 overnight deposit (real FRED SOFR fixing) +
#             N deposits at each SR3 reference-period end.
SR3_REFERENCE_PERIOD_YR = 0.25  # 3 months
usd_instruments = [
    CurveInstrument(
        type="deposit", maturity_years=1/365,
        rate=on_rate_used,
        day_count="ACT/360", payment_frequency=1.0,
    )
]
for _, row in sr3_strip.iterrows():
    t_end = float(row["expiry_years"]) + SR3_REFERENCE_PERIOD_YR
    if t_end <= 1/365:
        continue
    usd_instruments.append(CurveInstrument(
        type="deposit", maturity_years=t_end,
        rate=float(row["implied_rate"]),
        day_count="ACT/360", payment_frequency=1.0,
    ))

# Replace `bs` with a fresh bootstrapper instance for downstream risk-module use
bs = CurveBootstrapper(interpolation_method="log_linear")

print(f"  Curve built: {len(usd_curve.times)} nodes, max tenor = {usd_curve.times[-1]:.2f}Y")
print(f"  Strip rows: {len(sr3_strip)} (front + back contracts) | Build time: {t_curve*1000:.2f} ms")
print(f"  Overnight anchor: {on_rate_used*100:.4f}% from {on_rate_source}")
print(f"  Instruments derived from strip: {len(usd_instruments)} deposits "
      f"(1 overnight + {len(usd_instruments)-1} at SR3 reference-period ends)")
print("\n  Key USD rates (SOFR):")
for t in [0.5, 1, 2, 5]:
    if t <= usd_curve.times[-1]:
        print(f"    {t}Y zero: {usd_curve.zero_rate(t)*100:.3f}%  |  D({t}): {usd_curve.df(t):.6f}")
    else:
        print(f"    {t}Y zero: -- (beyond strip max tenor {usd_curve.times[-1]:.2f}Y)")

# --- 1c: Monotone Convex comparison ---
print("\n-- 1c: Monotone Convex vs Log-Linear --")
# Reuse the SAME bootstrap nodes as usd_curve (the per-segment SR3 forward
# bootstrap), only change the interpolation method.  Re-bootstrapping from
# usd_instruments with monotone-convex would silently treat each SR3 row as a
# long-dated [0, T] simple deposit, producing a spurious downward slope from
# the simple-to-continuous compounding penalty growing with tenor.
from module_a_curves.curve_bootstrapper import DiscountCurve as _DiscountCurve
import datetime as _dt
usd_curve_mc = _DiscountCurve(
    times=list(usd_curve.times),
    dfs=[usd_curve.df(t) for t in usd_curve.times],
    valuation_date=_dt.date.fromisoformat(VALUATION_DATE),
    interpolation_method="monotone_convex",
)

print(f"\n  {'Tenor':<8} {'LogLin Zero':>12} {'MonConv Zero':>13} {'Diff (bps)':>11}  |  "
      f"{'LogLin Fwd':>11} {'MonConv Fwd':>12} {'Diff (bps)':>11}")
print("  " + "-" * 95)
for t in [0.5, 1, 2, 3, 5, 7, 10, 20]:
    z_ll = usd_curve.zero_rate(t)
    z_mc = usd_curve_mc.zero_rate(t)
    f_ll = usd_curve.instantaneous_forward(t)
    f_mc = usd_curve_mc.instantaneous_forward(t)
    print(f"    {t:<6.1f} {z_ll*100:>11.4f}% {z_mc*100:>12.4f}% {(z_mc-z_ll)*10000:>+10.2f}  |  "
          f"{f_ll*100:>10.4f}% {f_mc*100:>11.4f}% {(f_mc-f_ll)*10000:>+10.2f}")

# Forward rate comparison plot
t_grid = np.linspace(0.1, min(usd_curve.times[-1] - 0.1, 29), 500)
fwd_ll = [usd_curve.instantaneous_forward(t) * 100 for t in t_grid]
fwd_mc = [usd_curve_mc.instantaneous_forward(t) * 100 for t in t_grid]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(t_grid, fwd_ll, label="Log-Linear", linewidth=2, color=JPM_COLORS["blue"])
axes[0].plot(t_grid, fwd_mc, label="Monotone Convex", linewidth=2, color=JPM_COLORS["red"], linestyle="--")
axes[0].set_xlabel("Maturity (years)")
axes[0].set_ylabel("Instantaneous Forward Rate (%)")
axes[0].set_title("USD OIS Forward Rates: Log-Linear vs Monotone Convex")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

diff_bps = [(mc - ll) * 100 for ll, mc in zip(fwd_ll, fwd_mc)]
axes[1].plot(t_grid, diff_bps, linewidth=2, color=JPM_COLORS["green"])
axes[1].axhline(y=0, color="black", linewidth=0.8)
axes[1].set_xlabel("Maturity (years)")
axes[1].set_ylabel("Difference (bps)")
axes[1].set_title("Forward Rate Difference (MonConv - LogLin)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "interpolation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/interpolation_comparison.png")

# --- Zero-coupon rate + discount factor curve ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

zr_ll = [usd_curve.zero_rate(t) * 100 for t in t_grid]
zr_mc = [usd_curve_mc.zero_rate(t) * 100 for t in t_grid]
df_ll = [usd_curve.df(t) for t in t_grid]
df_mc = [usd_curve_mc.df(t) for t in t_grid]

# SR3 reference windows: each contract covers [IMM, IMM + 0.25]
# expiry_years is stored as the IMM date (start of the 3-month reference period).
sr3_windows = [
    (float(row["expiry_years"]), float(row["expiry_years"]) + 0.25)
    for _, row in sr3_strip.iterrows()
]
sr3_windows = [(s, e) for s, e in sr3_windows if e <= t_grid[-1] + 0.05]

def _shade_sr3_windows(ax):
    """Shade SR3 reference periods + dashed lines at IMM (start) and IMM+3M (end)."""
    for i, (s, e) in enumerate(sr3_windows):
        # Alternate light shade so adjacent windows are distinguishable
        color = "#E8F0FF" if i % 2 == 0 else "#F5F8FF"
        ax.axvspan(s, e, color=color, alpha=0.6, zorder=0)
        # IMM start (solid grey, thinner)
        ax.axvline(x=s, color="grey", linewidth=0.5, alpha=0.5, zorder=1)
        # Ref end (slightly darker dashed)
        ax.axvline(x=e, color="grey", linewidth=0.5, alpha=0.5, linestyle="--", zorder=1)

# Left: zero rates
_shade_sr3_windows(axes[0])
axes[0].plot(t_grid, zr_ll, label="Log-Linear", linewidth=2,
             color=JPM_COLORS["blue"], zorder=3)
axes[0].plot(t_grid, zr_mc, label="Monotone Convex", linewidth=2,
             color=JPM_COLORS["red"], linestyle="--", zorder=3)
# Mark the FRED SOFR overnight anchor
axes[0].scatter([1/365], [on_rate_used * 100], color=JPM_COLORS["green"],
                s=70, zorder=5, edgecolors="black", linewidths=0.7,
                label=f"FRED SOFR fixing ({on_rate_used*100:.3f}%)")
# Mark front-contract IMM with annotation
if sr3_windows:
    imm_front, end_front = sr3_windows[0]
    axes[0].annotate(
        f"SR3M6 IMM\n(t={imm_front:.3f}y)",
        xy=(imm_front, axes[0].get_ylim()[0] if axes[0].get_ylim() else zr_ll[0]),
        xytext=(imm_front + 0.12, max(zr_ll) - 0.005),
        fontsize=8, color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.6),
    )
axes[0].set_xlabel("Maturity (years)")
axes[0].set_ylabel("Zero Rate (%)")
axes[0].set_title("USD OIS Zero-Coupon Rate Curve\n(shaded bands = SR3 reference windows)")
axes[0].legend(loc="best", framealpha=0.9)
axes[0].grid(True, alpha=0.25)

# Right: discount factors
_shade_sr3_windows(axes[1])
axes[1].plot(t_grid, df_ll, label="Log-Linear", linewidth=2,
             color=JPM_COLORS["blue"], zorder=3)
axes[1].plot(t_grid, df_mc, label="Monotone Convex", linewidth=2,
             color=JPM_COLORS["red"], linestyle="--", zorder=3)
axes[1].set_xlabel("Maturity (years)")
axes[1].set_ylabel("Discount Factor D(t)")
axes[1].set_title("USD OIS Discount Factor Curve\n(shaded bands = SR3 reference windows)")
axes[1].legend(loc="best", framealpha=0.9)
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(output_dir / "zero_discount_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/zero_discount_curves.png")

# --- 1d: FX Forward Curves via CIP -----------------------------------
# Real foreign OIS curves via free central-bank APIs; falls back to flat
# proxy only if the real-data fetch returns empty.
print("\n-- 1d: FX Forward Curves (Covered Interest Parity) --")

from module_a_curves.curve_bootstrapper import DiscountCurve, CurveInstrument
from module_a_curves.foreign_curve_bootstrap import bootstrap_foreign_ois_from_strip
from shared.databento_curve_loader import (
    fetch_estr_fst3_strip,
    fetch_estr_femp_strip,
    fetch_sonia_so3_strip,
    fetch_sonia_soa_strip,
)
from shared.exchange_rfr_loader import fetch_tona_strip, fetch_aonia_ib_strip
import datetime as dt

ccy_map = {"EUR/USD": "EUR", "GBP/USD": "GBP", "JPY/USD": "JPY", "AUD/USD": "AUD"}
fallback_flat = {"EUR/USD": 0.0290, "GBP/USD": 0.0450, "JPY/USD": 0.0010, "AUD/USD": 0.0435}

# Per-pair Tier-1 strip fetchers (futures-implied RFR forwards).  Each entry
# is a list of (label, fetcher, ref_period_yr).  We try them in order and
# concatenate any non-empty strips; if the combined strip is non-empty the
# pair gets a real-futures-bootstrapped curve.
strip_fetchers: dict[str, list[tuple[str, callable, float]]] = {
    "EUR/USD": [
        ("FST3", fetch_estr_fst3_strip, 0.25),
        ("FEMP", fetch_estr_femp_strip, 1.0 / 12.0),
    ],
    "GBP/USD": [
        ("SO3", fetch_sonia_so3_strip, 0.25),
        ("SOA", fetch_sonia_soa_strip, 1.0 / 12.0),
    ],
    "JPY/USD": [
        ("TY", fetch_tona_strip, 0.25),
    ],
    "AUD/USD": [
        ("IB", fetch_aonia_ib_strip, 1.0 / 12.0),
    ],
}

# Track the data tier each pair landed on (for the 1h plot legend).
foreign_curve_source: dict[str, str] = {}
foreign_curves = {}
# A 1-contract strip would otherwise win over a 6-tenor Tier-2 dict and
# produce a 1-segment flat curve (strictly worse).  Require at least 2
# contracts before preferring the futures-bootstrap path.
MIN_FUTURES_FOR_BOOTSTRAP = 2
for pair in G4_PAIRS:
    ccy = ccy_map[pair]

    # --- Tier-1: futures-strip bootstrap ---
    strip_pieces: list[pd.DataFrame] = []
    strip_labels: list[str] = []
    # Use the dominant ref-period across the contributing strips.  If both
    # 3M and 1M strips are present we prefer 3M (the tail of the curve is
    # the binding constraint); per-segment chaining handles month-vs-quarter
    # within the same DataFrame because each row is dispatched on its own
    # ref window, but the chained `bootstrap_foreign_ois_from_strip` uses a
    # single ref_period_yr for all rows.  In practice each pair has at most
    # one populated strip type today, so this collapses to a single ref.
    ref_period_for_pair = 0.25
    for label, fetcher, ref_yr in strip_fetchers.get(pair, []):
        try:
            piece = fetcher(VALUATION_DATE)
        except Exception:
            piece = None
        if piece is not None and not piece.empty:
            strip_pieces.append(piece)
            strip_labels.append(f"{label}={len(piece)}")
            ref_period_for_pair = ref_yr  # last non-empty wins; usually 3M

    combined_strip = (
        pd.concat(strip_pieces, ignore_index=True) if strip_pieces else pd.DataFrame()
    )

    if not combined_strip.empty and len(combined_strip) >= MIN_FUTURES_FOR_BOOTSTRAP:
        # Pull a foreign-overnight anchor from the Tier-2 dict's shortest-
        # tenor entry so the t=1/365 anchor matches the published central-
        # bank policy rate rather than the front-future implied forward.
        # Falls back to None (= front-strip rate) if missing.
        on_rate_for: float | None = None
        tier2_curve = loader.get_real_foreign_ois_curve(ccy, VALUATION_DATE)
        if tier2_curve:
            shortest = min(tier2_curve.keys())
            on_rate_for = float(tier2_curve[shortest])

        if pair == "AUD/USD":
            # AUD: combine 18-contract IB strip (1M-18M) with F2-derived
            # long-end CGS-adjusted points (2Y/3Y/5Y/10Y) for a denser curve.
            ib_instruments: list[CurveInstrument] = []
            if on_rate_for is not None:
                ib_instruments.append(CurveInstrument(
                    type="deposit", maturity_years=1 / 365, rate=on_rate_for,
                    day_count="ACT/360", payment_frequency=1.0,
                ))
            for _, row in combined_strip.iterrows():
                # IB futures reference an ~ 1-month accrual window starting
                # at expiry; place the implied rate at expiry + 1M as a
                # deposit anchor (matches the 1/12 ref_period_yr).
                ib_instruments.append(CurveInstrument(
                    type="deposit",
                    maturity_years=float(row["expiry_years"]) + 1.0 / 12.0,
                    rate=float(row["implied_rate"]),
                    day_count="ACT/360", payment_frequency=1.0,
                ))
            extra_instruments: list[CurveInstrument] = []
            long_end_tenors = [2.0, 3.0, 5.0, 10.0]
            if tier2_curve:
                for t in long_end_tenors:
                    if t in tier2_curve:
                        extra_instruments.append(CurveInstrument(
                            type="deposit", maturity_years=t,
                            rate=float(tier2_curve[t]),
                            day_count="ACT/360", payment_frequency=1.0,
                        ))
            all_aud_instruments = ib_instruments + extra_instruments
            bs_for = CurveBootstrapper(interpolation_method="log_linear")
            foreign_curves[pair] = bs_for.bootstrap(all_aud_instruments, VALUATION_DATE)
            n_ib = len(combined_strip)
            n_extra = len(extra_instruments)
            foreign_curve_source[pair] = (
                f"REAL FUTURES + F2 LONG-END ({n_ib} IB + {n_extra} F2)"
            )
            print(
                f"  {pair} curve: REAL FUTURES + F2 LONG-END "
                f"({n_ib} IB contracts + {n_extra} F2-derived nodes)"
            )
            continue

        foreign_curves[pair] = bootstrap_foreign_ois_from_strip(
            combined_strip,
            valuation_date=VALUATION_DATE,
            overnight_rate=on_rate_for,
            ref_period_yr=ref_period_for_pair,
        )
        n_total = len(combined_strip)
        sources_str = " / ".join(strip_labels)
        foreign_curve_source[pair] = f"REAL FUTURES ({sources_str})"
        print(f"  {pair} curve: REAL FUTURES ({n_total} contracts: {sources_str})")
        continue

    # --- Tier-2: FRED / RBA term structure ---
    real_curve = loader.get_real_foreign_ois_curve(ccy, VALUATION_DATE)
    if real_curve and len(real_curve) >= 2:
        instruments = []
        for tenor, rate in sorted(real_curve.items()):
            inst_type = "deposit" if tenor < 0.5 else "swap"
            instruments.append(CurveInstrument(
                type=inst_type, maturity_years=tenor, rate=rate,
                day_count="ACT/360", payment_frequency=1.0,
            ))
        bs_for = CurveBootstrapper(interpolation_method="log_linear")
        foreign_curves[pair] = bs_for.bootstrap(instruments, VALUATION_DATE)
        foreign_curve_source[pair] = f"REAL TERM ({len(real_curve)} tenors)"
        print(f"  {pair} curve: REAL TERM ({len(real_curve)} tenors from FRED Tier-2)")
        continue

    # --- Tier-3: flat fallback ---
    rate = fallback_flat[pair]
    flat_instruments = [
        CurveInstrument(type="deposit", maturity_years=1/360, rate=rate, day_count="ACT/360"),
        CurveInstrument(type="swap",    maturity_years=1.0,   rate=rate,         day_count="ACT/360", payment_frequency=1.0),
        CurveInstrument(type="swap",    maturity_years=2.0,   rate=rate + 0.001, day_count="ACT/360", payment_frequency=1.0),
        CurveInstrument(type="swap",    maturity_years=5.0,   rate=rate + 0.003, day_count="ACT/360", payment_frequency=1.0),
        CurveInstrument(type="swap",    maturity_years=10.0,  rate=rate + 0.005, day_count="ACT/360", payment_frequency=1.0),
    ]
    bs_for = CurveBootstrapper(interpolation_method="log_linear")
    foreign_curves[pair] = bs_for.bootstrap(flat_instruments, VALUATION_DATE)
    foreign_curve_source[pair] = "FALLBACK FLAT"
    print(f"  {pair} curve: FALLBACK FLAT ({ccy} real-data fetch returned empty)")

# Build FX forward curves
fx_forward_curves = {}
print(f"\n  {'Pair':<10} {'Spot':>8} {'1M Fwd':>10} {'3M Fwd':>10} {'1Y Fwd':>10} {'1Y Pts (pips)':>14}")
print("  " + "-" * 60)
for pair in G4_PAIRS:
    spot = fx_spots.get(pair, 1.0)
    fwd_curve = FXForwardCurve(spot, usd_curve, foreign_curves[pair], pair)
    fx_forward_curves[pair] = fwd_curve
    fwd_1m = fwd_curve.forward(1/12)
    fwd_3m = fwd_curve.forward(3/12)
    fwd_1y = fwd_curve.forward(1.0)
    pts_1y = fwd_curve.forward_points(1.0)
    pip_factor = 10000 if "JPY" not in pair else 100
    print(f"    {pair:<10} {spot:>8.4f} {fwd_1m:>10.4f} {fwd_3m:>10.4f} {fwd_1y:>10.4f} {pts_1y*pip_factor:>+13.1f}")

# --- 1e: Cross-Currency Basis (CIP Deviations from CME futures) ------
print("\n-- 1e: Cross-Currency Basis (CIP Deviations) --")
from shared.databento_curve_loader import fetch_back_month_fx_curve

ticker_for_pair = {"EUR/USD": "6E", "GBP/USD": "6B", "JPY/USD": "6J", "AUD/USD": "6A"}

# Cache the CME-derived basis curves so Section 1f's surface plot can use them
# as the primary source, falling back to DTCC PPD only where CME is missing.
cme_basis_curves: dict[str, pd.DataFrame] = {}
for pair in G4_PAIRS:
    futures_df = fetch_back_month_fx_curve(ticker_for_pair[pair], VALUATION_DATE, n_contracts=5)
    if futures_df is not None and not futures_df.empty:
        ccb_pair = CrossCurrencyBasis.from_cme_futures(
            spot=fx_spots[pair],
            usd_curve=usd_curve,
            foreign_curve=foreign_curves[pair],
            futures_df=futures_df,
            pair=pair,
        )
        basis_df = ccb_pair.basis_curve(pair)
        source_tag = "REAL"
    else:
        # Fallback: synthetic generator
        ccb_pair = CrossCurrencyBasis()
        basis_df = ccb_pair.basis_curve(pair)
        source_tag = "synthetic"

    cme_basis_curves[pair] = basis_df
    basis_str = "  ".join(f"{row['tenor']}: {row['basis_bps']:+.1f}" for _, row in basis_df.iterrows())
    print(f"    {pair} ({source_tag}): {basis_str}")

# --- 1f: FX Swap Mechanics (forward points, T/N, turn-of-year, basis arb) ----
print("\n-- 1f: FX Swap Mechanics --")

from module_b_trading.fx_swap_mechanics import (
    forward_points, tn_roll_points, apply_turn_overlay, pip_factor,
)
# DTCC SDR loader retained for ad-hoc CME-vs-DTCC validation (see
# `output/xccy_basis_cme_vs_dtcc.csv`); no longer used by Step 1f, which is
# now CME-only.

# Worked-example check: USD/JPY 3M forward points
fwd, pts = forward_points(
    spot=150.00, rate_base=0.045, rate_quote=0.0025,
    tenor_years=0.25, pair="USD/JPY",
)
print(f"  USD/JPY 3M (textbook): F={fwd:.2f}, points={pts:+.0f} pips (JPY at premium)")

# Live forward points using project curves (real where available)
print("\n  Live G4 swap points (real curves):")
print(f"  {'Pair':<10} {'Spot':>10} {'3M Fwd':>10} {'3M Pts':>10}")
for pair in G4_PAIRS:
    spot = fx_spots[pair]
    foreign_curve = foreign_curves[pair]
    r_base = -np.log(foreign_curve.df(0.25)) / 0.25
    r_quote = -np.log(usd_curve.df(0.25)) / 0.25
    fwd_p, pts_p = forward_points(spot, r_base, r_quote, 0.25, pair)
    print(f"  {pair:<10} {spot:>10.4f} {fwd_p:>10.4f} {pts_p:>+9.1f}")

# T/N roll example
tn = tn_roll_points(spot=150.00, rate_base=0.045, rate_quote=0.0025, pair="USD/JPY")
print(f"\n  USD/JPY T/N roll: {tn:+.2f} pips (1-day USD-funding cost)")

# Turn-of-year overlay
toy_pts = apply_turn_overlay(pts, "2026-12-29", "2027-01-05", "USD/JPY")
print(f"  USD/JPY 1W swap, year-end window: {toy_pts:+.1f} pips (vs base {pts:+.1f})")

# Cross-currency basis term structure (CME-derived only, at actual IMM tenors).
# `cme_basis_curves[pair]` was built in 1e from `CrossCurrencyBasis.from_cme_futures`
# at whatever IMM tenors the cache reaches (typically ~2M / 5M / 8M / 11M / 1.1Y
# from val_date). No DTCC fallback, no interpolation onto standard tenors —
# the table mirrors what the CIP comparison consumes downstream.
print("\n  Cross-currency basis term structure (bps, CME-derived at IMM tenors):")
basis_curves = {}
for pair in G4_PAIRS:
    cme_df = cme_basis_curves.get(pair)
    if cme_df is None or cme_df.empty:
        basis_curves[pair] = pd.DataFrame(columns=["tenor_label", "basis_bps"]).set_index("tenor_label")
        print(f"  {pair:<10} (no CME basis data)")
        continue
    df = cme_df[["tenor", "maturity_years", "basis_bps"]].copy()
    df.columns = ["tenor_label", "maturity_years", "basis_bps"]
    basis_curves[pair] = df.set_index("tenor_label")
    cells = [f"{row['tenor_label']}: {row['basis_bps']:+7.2f}" for _, row in df.iterrows()]
    print(f"  {pair:<10} " + "   ".join(cells))


# --- 1g: Pipeline Visualizations -----------------------------------
print("\n-- 1g: Pipeline Visualizations --")

# Plot 1: foreign OIS curves
fig, ax = plt.subplots(figsize=(10, 6))
t_grid = np.logspace(-2, 1, 200)  # 0.01 to 10 years
labels_sources = [
    ("USD (SOFR strip)", usd_curve, JPM_COLORS["blue"]),
    (f"EUR ({foreign_curve_source['EUR/USD']})", foreign_curves["EUR/USD"], JPM_COLORS["red"]),
    (f"GBP ({foreign_curve_source['GBP/USD']})", foreign_curves["GBP/USD"], JPM_COLORS["green"]),
    (f"JPY ({foreign_curve_source['JPY/USD']})", foreign_curves["JPY/USD"], JPM_COLORS.get("orange", "#FFA500")),
    (f"AUD ({foreign_curve_source['AUD/USD']})", foreign_curves["AUD/USD"], JPM_COLORS.get("purple", "#9966CC")),
]
for label, curve, color in labels_sources:
    zero_rates = [curve.zero_rate(t) * 100 for t in t_grid]
    ax.plot(t_grid, zero_rates, label=label, linewidth=2, color=color, marker="o", markersize=3)
# linear x-axis with explicit tick labels at standard tenors
ax.set_xlim(0, 10.5)
ax.set_xticks([0, 0.25, 0.5, 1, 2, 5, 10])
ax.set_xticklabels(["0", "3M", "6M", "1Y", "2Y", "5Y", "10Y"])
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("Zero rate (%)")
ax.set_title("OIS Term Structures (Real Data)")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(output_dir / "foreign_ois_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/foreign_ois_curves.png")

# Plot 2: 3D xccy basis surface (CME-derived at IMM tenors).
# Each pair's `basis_curves[pair]` is indexed by IMM tenor label and carries a
# `maturity_years` column. Use the union of maturity years across pairs so the
# surface has one common Y axis; missing cells are NaN (rendered as gaps).
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
pair_labels = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD"]
union_T = sorted({float(t) for p in pair_labels
                  for t in (basis_curves[p]["maturity_years"].tolist()
                            if "maturity_years" in basis_curves[p].columns else [])})
basis_grid = np.full((len(pair_labels), len(union_T)), np.nan)
for i, pair in enumerate(pair_labels):
    df_basis = basis_curves[pair]
    if "maturity_years" not in df_basis.columns:
        continue
    for _, row in df_basis.iterrows():
        T = float(row["maturity_years"])
        if T in union_T:
            basis_grid[i, union_T.index(T)] = float(row["basis_bps"])
X, Y = np.meshgrid(np.arange(len(pair_labels)), np.array(union_T))
Z = basis_grid.T  # shape (n_tenors, n_pairs)
surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="k", linewidth=0.3, alpha=0.9)
ax.set_xticks(np.arange(len(pair_labels)))
ax.set_xticklabels(pair_labels)
ax.set_xlabel("Pair")
ax.set_ylabel("Tenor (years, IMM-spaced)")
ax.set_zlabel("Basis (bps)")
ax.set_title("Cross-Currency Basis Term Structure (CME-derived at IMM tenors, G4 vs USD)")
fig.colorbar(surf, ax=ax, shrink=0.6, label="bps")
plt.tight_layout()
fig.savefig(output_dir / "xccy_basis_surface.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/xccy_basis_surface.png")

# Plot 3: forward points heatmap
import seaborn as sns
tenors = {"1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}
fp_grid = np.zeros((len(G4_PAIRS), len(tenors)))
for i, pair in enumerate(G4_PAIRS):
    spot = fx_spots[pair]
    foreign_curve = foreign_curves[pair]
    for j, (label, t) in enumerate(tenors.items()):
        r_base = -np.log(foreign_curve.df(t)) / t
        r_quote = -np.log(usd_curve.df(t)) / t
        _, pips = forward_points(spot, r_base, r_quote, t, pair)
        fp_grid[i, j] = pips
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    fp_grid, annot=True, fmt="+.1f",
    xticklabels=list(tenors.keys()), yticklabels=G4_PAIRS,
    cmap="RdYlGn", center=0,
    cbar_kws={"label": "Forward points (pips)"},
    ax=ax,
)
ax.set_title("CIP-Implied Forward Points (G4 x Tenors, Real Curves)")
ax.set_xlabel("Tenor")
ax.set_ylabel("Pair")
plt.tight_layout()
fig.savefig(output_dir / "forward_points_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/forward_points_heatmap.png")


###############################################################################
# STEP 2: FX Pricing & Risk
###############################################################################

print("\n" + "=" * 70)
print("  STEP 2: FX Pricing & Risk")
print("  (FX Swap Portfolio, Spot Delta, Cross-Gamma, Scenarios, VaR)")
print("=" * 70)

from module_b_trading.fx_pricer import FXPricer, FXSwapSpec
from module_b_trading.risk_analytics import FXRiskAnalytics
from module_b_trading.market_forward import MarketForwardCurve

# Build pricers per pair, attaching a CME-quoted MarketForwardCurve so
# fx_swap_npv marks against exchange-cleared forwards (with CIP fallback for
# tenors beyond the cached IMM grid).
pricers = {}
market_curves = {}
for pair in G4_PAIRS:
    spot = fx_spots.get(pair, 1.0)
    cip_pricer = FXPricer(usd_curve, foreign_curves[pair], spot, pair)
    futures_grid = fetch_back_month_fx_curve(
        ticker_for_pair[pair], VALUATION_DATE, n_contracts=5,
    )
    market_curves[pair] = MarketForwardCurve(
        futures_grid, spot, cip_fallback=cip_pricer.forward_price,
    )
    pricers[pair] = FXPricer(usd_curve, foreign_curves[pair], spot, pair,
                             market_curve=market_curves[pair])

# --- 2a: Portfolio Pricing ---
print("\n-- 2a: FX Swap Portfolio Pricing (CME-quoted forwards) --")
portfolio = [
    FXSwapSpec(10_000_000, 1.0, market_curves["EUR/USD"].f_market(1.0)[0] - 0.005, "EUR/USD", "buy_base"),
    FXSwapSpec(5_000_000, 0.5, market_curves["GBP/USD"].f_market(0.5)[0] + 0.002, "GBP/USD", "sell_base"),
    FXSwapSpec(500_000_000, 1.0, market_curves["JPY/USD"].f_market(1.0)[0] * 1.01, "JPY/USD", "buy_base"),
    FXSwapSpec(8_000_000, 0.25, market_curves["AUD/USD"].f_market(0.25)[0] - 0.003, "AUD/USD", "buy_base"),
]

print(f'  {"Swap":<35} {"NPV (USD)":>14} {"Fwd Mkt":>10} {"Fwd CIP":>10} {"Src":>10} {"Δ(bps)":>8}')
print("  " + "=" * 91)
total_npv = 0
for swap in portfolio:
    pricer = pricers[swap.pair]
    npv = pricer.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
    npv_signed = npv * swap.sign()
    fwd_mkt, src = market_curves[swap.pair].f_market(swap.maturity_years)
    fwd_cip = pricer.forward_price(swap.maturity_years)
    diff_bps = (fwd_mkt / fwd_cip - 1.0) * 1e4 if fwd_cip else 0.0
    dir_label = "BUY" if swap.direction == "buy_base" else "SELL"
    total_npv += npv_signed
    ccy = swap.pair.split("/")[0]
    print(f"  {dir_label} {swap.notional/1e6:.0f}MM {ccy} {swap.maturity_years:.2f}Y"
          f"{'':>14} {npv_signed:>14,.0f} {fwd_mkt:>10.4f} {fwd_cip:>10.4f} "
          f"{src:>10} {diff_bps:>+8.2f}")
print("  " + "=" * 91)
print(f'  {"TOTAL PORTFOLIO NPV":<35} {total_npv:>14,.0f}')
print(f'  Δ(bps) = (Fwd_CME / Fwd_CIP − 1) × 1e4. Sign of Δ is the cross-currency basis priced into the')
print(f'  futures relative to the OIS-derived CIP forward. Source legend: CME=interp between IMMs,')
print(f'  CME-stub=spot→first-IMM front stub, CIP=fallback for T beyond cached grid.')

# --- 2b: Risk Analytics ---
print("\n-- 2b: Risk (Spot Delta, Rate Delta, Tenor-Matched Hedge) --")
# Pass per-pair IMM grids so best_futures_hedge picks the contract whose
# expiry is closest to each swap's maturity (kills roll risk, aligns the
# hedge's basis sensitivity with the swap).
futures_grids = {p: fetch_back_month_fx_curve(ticker_for_pair[p], VALUATION_DATE, n_contracts=5)
                 for p in G4_PAIRS}
ra = FXRiskAnalytics(bs, {"USD": usd_instruments}, VALUATION_DATE, fx_spots,
                     futures_grids=futures_grids)

print(f'  {"Swap":<30} {"Spot Delta (USD)":>16} {"Hedge Contract":>16} {"#":>5}')
print("  " + "-" * 70)
hedge_results = {}
for swap in portfolio:
    pricer = pricers[swap.pair]
    delta = pricer.delta(swap.notional, swap.maturity_years) * swap.sign()
    hedge = ra.best_futures_hedge(swap)
    dir_label = "BUY" if swap.direction == "buy_base" else "SELL"
    ccy = swap.pair.split("/")[0]
    sign = "+" if hedge["direction"] == "long" else "-"
    print(f"  {dir_label} {swap.notional/1e6:.0f}MM {ccy} {swap.maturity_years:.2f}Y"
          f"{'':>10} {delta:>16,.0f} {hedge['contract']:>16} {sign}{hedge['n_contracts']:>4}")
    hedge_results[swap.pair] = hedge

# --- 2c: Cross-pair correlation ---
# Fetch 2y of daily FX closes from yfinance and convert to log returns. Both
# the correlation matrix and VaR vols are estimated from this same window so
# Σ = diag(σ) · ρ · diag(σ) is internally consistent.
from shared.yfinance_loader import fetch_fx_daily

fx_prices = fetch_fx_daily(pairs=G4_PAIRS, period="2y")
if fx_prices is not None and len(fx_prices) >= 20:
    returns_history = np.log(fx_prices).diff().dropna()
    print(f"\n-- 2c: Cross-Pair Correlation Matrix ({len(returns_history)}d realized) --")
else:
    returns_history = None
    print("\n-- 2c: Cross-Pair Correlation Matrix (defaults — yfinance unavailable) --")

corr = ra.correlation_matrix(returns_history)
print(corr.to_string(float_format="{:.3f}".format))

# --- 2d: Parametric VaR ---
print("\n-- 2d: Parametric VaR (99%, 1-day) --")
var_result = ra.var_portfolio(portfolio, confidence=0.99, returns_history=returns_history)
if returns_history is not None:
    ann_vols = returns_history.std() * np.sqrt(252) * 100
    vol_str = ", ".join(f"{p}={ann_vols[p]:.1f}%" for p in returns_history.columns if p in ann_vols)
    print(f"  Realized annualised vols: {vol_str}")
print(f"  Portfolio 1-day 99% VaR: ${var_result:,.0f}")

# Risk ladder chart
fig, ax = plt.subplots(figsize=(10, 5))
pairs_in_port = [s.pair for s in portfolio]
deltas = []
for swap in portfolio:
    pricer = pricers[swap.pair]
    d = pricer.delta(swap.notional, swap.maturity_years) * swap.sign()
    deltas.append(d)
colors = [JPM_COLORS["blue"] if v >= 0 else JPM_COLORS["red"] for v in deltas]
ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.8, tick_label=pairs_in_port)
ax.set_ylabel("Spot Delta (USD)")
ax.set_title(f"FX Portfolio Spot Delta by Pair (VaR = ${var_result:,.0f})")
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(y=0, color="black", linewidth=0.8)
plt.tight_layout()
fig.savefig(output_dir / "fx_delta_ladder.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/fx_delta_ladder.png")


###############################################################################
# STEP 3: Fix Alpha Signals & Quoting
###############################################################################

print("\n" + "=" * 70)
print("  STEP 3: Fix Alpha Signals & Quoting")
print("  (W-Shaped Fix Pattern, Carry/Momentum/MR, Win Model, Optimizer)")
print("=" * 70)

from module_b_trading.fix_alpha_signals import FixAlphaModel, CompositeAlphaModel
from module_b_trading.rfq_generator import FXRFQGenerator
from module_b_trading.win_probability import FXWinProbabilityModel
from module_b_trading.quote_optimizer import FXQuoteOptimizer

# --- 3a: W-Shaped Fix Pattern ---
print("\n-- 3a: W-Shaped Fix Pattern (Krohn, Mueller & Whelan 2024) --")
fix_model = FixAlphaModel()

# Try REAL Yahoo Finance data first, fall back to synthetic
intraday_data, data_source = fix_model.get_intraday_data(pairs=G4_PAIRS, prefer_real=True)
# Count unique calendar dates — robust to caches that only cover fix windows
# rather than the full 96 buckets per day.
n_days_approx = max(1, intraday_data["timestamp"].dt.date.nunique())
if data_source == "cme_aggregated":
    print(f"  REAL CME DATA: {len(intraday_data)} 15-min observations (~{n_days_approx} days from CME futures)")
elif data_source.startswith("yahoo"):
    print(f"  REAL DATA from {data_source}: {len(intraday_data)} observations (~{n_days_approx} days)")
    print(f"  Date range: {intraday_data['timestamp'].min()} -> {intraday_data['timestamp'].max()}")
else:
    print(f"  Synthetic data: {len(intraday_data)} observations ({n_days_approx} days)")
    print(f"  (install yfinance for real data: pip install yfinance)")

# Compute the W-pattern: average return by time-of-day bucket
w_pattern = fix_model.compute_w_pattern(intraday_data)
print(f"\n  W-Pattern Summary (EUR/USD, returns in bps around fixes):")

# Build mid_price series from returns for compute_fix_returns
# Start from spot rates and accumulate 15-min returns
price_data_frames = []
for pair in G4_PAIRS:
    pair_mask = intraday_data["pair"] == pair
    pair_returns = intraday_data[pair_mask].copy()
    start_price = fx_spots.get(pair, 1.0)
    pair_returns["mid_price"] = start_price * (1 + pair_returns["return_15m"]).cumprod()
    price_data_frames.append(pair_returns[["timestamp", "pair", "mid_price"]])
price_history = pd.concat(price_data_frames, ignore_index=True)

fix_returns = fix_model.compute_fix_returns(price_history)

# Display fix returns -- show per-fix, per-window
print(f"\n  {'Fix':<10} {'Window':<15} {'Mean (bps)':>12} {'t-stat':>8} {'n_days':>8}")
print("  " + "-" * 58)
# Filter to London fix and EUR/USD for main display
london_eur = fix_returns[
    (fix_returns.get("fix_name", fix_returns.get("fix", "")) == "london") &
    (fix_returns.get("pair", "") == "EUR/USD")
] if "fix_name" in fix_returns.columns else fix_returns.head(6)
for _, row in london_eur.iterrows():
    fix_name = row.get("fix_name", row.get("fix", "london"))
    window = row.get("window", "")
    mean_r = row.get("mean_return_bps", 0)
    t_stat = row.get("t_stat", 0)
    n_days = row.get("n_days", 0)
    print(f"    {fix_name:<10} {window:<15} {mean_r:>+11.2f} {t_stat:>+7.2f} {n_days:>8}")

print("\n  Key finding: Pre-fix returns are NEGATIVE (USD appreciation)")
print("  Post-fix returns are POSITIVE (USD depreciation) = the W-shape")

# Plot W-shape pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: intraday cumulative return for EUR/USD
# Compute mean return per 15-min bucket from intraday_data
eur_intraday = intraday_data[intraday_data["pair"] == "EUR/USD"].copy()
eur_intraday["bucket"] = eur_intraday["timestamp"].dt.hour * 4 + eur_intraday["timestamp"].dt.minute // 15
bucket_means = eur_intraday.groupby("bucket")["return_15m"].mean() * 10000  # to bps
cum_returns = bucket_means.sort_index().cumsum()
axes[0].plot(cum_returns.index, cum_returns.values, linewidth=2, color=JPM_COLORS["blue"])
# Truncate x-axis to the actual data range so the plot is honest about coverage
b_lo = int(bucket_means.index.min()) - 2
b_hi = int(bucket_means.index.max()) + 2
axes[0].set_xlim(b_lo, b_hi)
# Re-label x ticks as HH:MM derived from 15-min buckets
tick_buckets = list(range(b_lo, b_hi + 1, 2))
tick_labels = [f"{(b * 15) // 60:02d}:{(b * 15) % 60:02d}" for b in tick_buckets]
axes[0].set_xticks(tick_buckets)
axes[0].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
axes[0].set_xlabel("Time of day (UTC, 15-min buckets)")
axes[0].set_ylabel("Cumulative Return (bps)")
axes[0].set_title("EUR/USD Cumulative Return (London/NY Window, 15-min)")
# Mark fix times only if they fall inside the visible window
out_of_range = []
for fix_name, bucket_idx in [("Tokyo", 3), ("ECB", 53), ("London", 64)]:
    if b_lo <= bucket_idx <= b_hi:
        axes[0].axvline(x=bucket_idx, color=JPM_COLORS["red"], linestyle="--", alpha=0.7, label=f"{fix_name} fix")
    else:
        out_of_range.append(fix_name)
if out_of_range:
    # Surface the limited coverage in the legend itself
    axes[0].plot([], [], " ", label=f"{', '.join(out_of_range)} fix: fix-window coverage only")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right: fix return windows bar chart (London fix, EUR/USD)
london_eur_data = fix_returns[
    (fix_returns["fix_name"] == "london") & (fix_returns["pair"] == "EUR/USD")
] if "fix_name" in fix_returns.columns and "pair" in fix_returns.columns else fix_returns.head(6)
if len(london_eur_data) > 0:
    windows = london_eur_data["window"].values
    returns_bps = london_eur_data["mean_return_bps"].values
    bar_colors = [JPM_COLORS["red"] if v < 0 else JPM_COLORS["green"] for v in returns_bps]
    axes[1].bar(range(len(windows)), returns_bps, color=bar_colors, alpha=0.8)
    axes[1].set_xticks(range(len(windows)))
    axes[1].set_xticklabels(windows, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("EUR/USD Returns Around London Fix (bps)")
    axes[1].set_ylabel("Mean Return (bps)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="black", linewidth=0.8)

plt.tight_layout()
fig.savefig(output_dir / "w_shape_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/w_shape_pattern.png")

# --- Small multiples: all G4 pairs side-by-side for cross-pair comparison ---
# Same axes (cumulative return curve top, fix-window bars bottom) per pair,
# with shared y-limits so magnitudes are visually comparable.
fig_sm, axes_sm = plt.subplots(2, len(G4_PAIRS), figsize=(4 * len(G4_PAIRS), 8), sharey="row")

# Pre-compute per-pair series so we can set shared y-limits in one pass
cum_by_pair = {}
bars_by_pair = {}
for pair in G4_PAIRS:
    p_intraday = intraday_data[intraday_data["pair"] == pair].copy()
    p_intraday["bucket"] = p_intraday["timestamp"].dt.hour * 4 + p_intraday["timestamp"].dt.minute // 15
    p_bucket_means = p_intraday.groupby("bucket")["return_15m"].mean() * 10000  # bps
    cum_by_pair[pair] = p_bucket_means.sort_index().cumsum()

    p_london = fix_returns[
        (fix_returns["fix_name"] == "london") & (fix_returns["pair"] == pair)
    ] if {"fix_name", "pair"} <= set(fix_returns.columns) else fix_returns.head(6)
    bars_by_pair[pair] = p_london

# Determine x-window from union of buckets across all pairs
all_buckets = sorted(set().union(*[set(s.index) for s in cum_by_pair.values()]))
sm_b_lo = all_buckets[0] - 2
sm_b_hi = all_buckets[-1] + 2
sm_tick_buckets = list(range(sm_b_lo, sm_b_hi + 1, 2))
sm_tick_labels = [f"{(b * 15) // 60:02d}:{(b * 15) % 60:02d}" for b in sm_tick_buckets]

for col_idx, pair in enumerate(G4_PAIRS):
    ax_top = axes_sm[0, col_idx]
    ax_bot = axes_sm[1, col_idx]

    # Top row: cumulative return curve
    cum = cum_by_pair[pair]
    ax_top.plot(cum.index, cum.values, linewidth=2, color=JPM_COLORS["blue"])
    ax_top.set_xlim(sm_b_lo, sm_b_hi)
    ax_top.set_xticks(sm_tick_buckets)
    ax_top.set_xticklabels(sm_tick_labels, rotation=45, ha="right", fontsize=7)
    ax_top.axhline(y=0, color="black", linewidth=0.6, alpha=0.5)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(pair, fontsize=11, fontweight="bold")
    if col_idx == 0:
        ax_top.set_ylabel("Cumulative Return (bps)")

    # Mark fix lines that fall inside the visible window
    for fix_name, bucket_idx in [("Tokyo", 3), ("ECB", 53), ("London", 64)]:
        if sm_b_lo <= bucket_idx <= sm_b_hi:
            ax_top.axvline(x=bucket_idx, color=JPM_COLORS["red"], linestyle="--",
                           alpha=0.7, linewidth=1)

    # Bottom row: fix-window bar chart
    p_london = bars_by_pair[pair]
    if len(p_london) > 0:
        windows = p_london["window"].values
        returns_bps = p_london["mean_return_bps"].values
        bar_colors = [JPM_COLORS["red"] if v < 0 else JPM_COLORS["green"] for v in returns_bps]
        ax_bot.bar(range(len(windows)), returns_bps, color=bar_colors, alpha=0.8)
        ax_bot.set_xticks(range(len(windows)))
        ax_bot.set_xticklabels(windows, rotation=45, ha="right", fontsize=7)
        ax_bot.axhline(y=0, color="black", linewidth=0.6)
        ax_bot.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_bot.set_ylabel("Mean Return (bps)")
        ax_bot.set_xlabel("Window around London fix")

fig_sm.suptitle(
    "W-Shape Pattern Across G4 Pairs (top: cumulative intraday; bottom: London-fix windows)",
    fontsize=12, fontweight="bold", y=1.00,
)
plt.tight_layout()
fig_sm.savefig(output_dir / "w_shape_pattern_g4.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/w_shape_pattern_g4.png (4-panel small multiples)")

# --- 3b: Composite Alpha Signals ---
print("\n-- 3b: Composite Alpha Signals (Fix + Carry + Momentum + MR) --")
rates_history = loader.fetch_history(start="2025-09-01", end=VALUATION_DATE)

# If FRED failed (all synthetic GBM), try yfinance daily as fallback
if rates_history is not None and len(rates_history) < 50:
    try:
        from shared.yfinance_loader import fetch_fx_daily
        yf_daily = fetch_fx_daily(pairs=G4_PAIRS, start="2025-09-01", end=VALUATION_DATE)
        if yf_daily is not None and len(yf_daily) > 50:
            rates_history = yf_daily
            print(f"  Daily FX history from Yahoo Finance: {len(rates_history)} days")
    except Exception:
        pass
alpha_model = CompositeAlphaModel(fix_wt=0.50, carry_wt=0.25, momentum_wt=0.15, mr_wt=0.10, max_skew_pips=3.0)

# Use a representative time: 15:30 UTC (30 min before London fix)
test_time = datetime(2026, 3, 5, 15, 30)
signals = alpha_model.compute_signals(rates_history, intraday_data, current_utc_time=test_time)

print(f"  Signals at {test_time.strftime('%H:%M')} UTC (30min before London fix):")
for key, val in signals.items():
    if isinstance(val, dict):
        for k, v in val.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                print(f"    {key}.{k}: {v:+.4f}")
    elif isinstance(val, (int, float)) and not np.isnan(val):
        print(f"    {key}: {val:+.4f}")

# --- 3c: Win Probability Model ---
print("\n-- 3c: Win Probability Model --")
gen = FXRFQGenerator()
# Use recent date range that overlaps with yfinance intraday data (last ~60 days)
# so that markout analysis can use REAL prices for P&L computation
from datetime import date
_rfq_end = date.today().isoformat()
_rfq_start = (date.today() - timedelta(days=55)).isoformat()
rfqs = gen.generate(2000, fx_spots, _rfq_start, _rfq_end, seed=42)
print(f"  RFQ date range: {_rfq_start} -> {_rfq_end} (overlaps with yfinance 1h data)")
wpm = FXWinProbabilityModel()
metrics = wpm.train(rfqs)
auc = metrics.get('auc_test', metrics.get('auc', 0))
accuracy = metrics.get('accuracy_test', metrics.get('accuracy', 0))
print(f"  Win Model: AUC={auc:.3f}, Accuracy={accuracy:.3f}")

eval_metrics = wpm.evaluate(rfqs)
cal_key = "calibration_table" if "calibration_table" in eval_metrics else "calibration"
if cal_key in eval_metrics:
    cal_data = eval_metrics[cal_key]
    print(f"\n  Decile Calibration (predicted vs actual hit rate):")
    print(f"    {'Bucket':<14} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
    print("    " + "-" * 44)
    if isinstance(cal_data, pd.DataFrame):
        for _, row in cal_data.iterrows():
            bkt = row.get("bucket", row.get("bin", ""))
            pred = row.get("mean_predicted", row.get("predicted_mean", 0))
            actual = row.get("mean_actual", row.get("actual_mean", 0))
            cnt = row.get("count", 0)
            print(f"    {str(bkt):<14} {pred:>10.3f} {actual:>10.3f} {cnt:>8}")
    elif isinstance(cal_data, list):
        for row in cal_data:
            bkt = row.get("bucket", row.get("bin", ""))
            pred = row.get("mean_predicted", row.get("predicted_mean", 0))
            actual = row.get("mean_actual", row.get("actual_mean", 0))
            cnt = row.get("count", 0)
            print(f"    {str(bkt):<14} {pred:>10.3f} {actual:>10.3f} {cnt:>8}")
print(f"  Brier Score: {eval_metrics.get('brier_score', 0):.4f}")
print(f"  AUC: {eval_metrics.get('auc', auc):.3f}")

# Drift detection
split = len(rfqs) // 2
drift = wpm.detect_drift(rfqs.iloc[split:], rfqs.iloc[:split])
max_psi = drift.get("max_psi", drift.get("prediction_psi", 0))
print(f"\n  Drift Detection:")
print(f"    Max Feature PSI: {max_psi:.4f}", end="")
if max_psi < 0.10:
    print("  (stable)")
elif max_psi < 0.20:
    print("  (moderate drift)")
else:
    print("  (significant drift -- retrain)")
print(f"    Recommendation: {drift.get('recommendation', 'N/A')}")

# --- 3d: Fix-Aware Quote Optimizer ---
print("\n-- 3d: Fix-Aware Quote Optimizer (with directional alpha) --")
optimizer = FXQuoteOptimizer(wpm, fix_model, lambda_risk=1e-6)
subset = rfqs.head(100)

# Demonstrate quoting at different times. The alpha view comes from the
# CompositeAlphaModel built earlier in 3b.
pre_fix_time = datetime(2026, 3, 5, 15, 30)  # 30min before London fix
at_fix_time = datetime(2026, 3, 5, 16, 1)    # 1min after London fix print
off_fix_time = datetime(2026, 3, 5, 10, 0)   # no fix nearby

# composite_skew_pips at each scenario time
alpha_pre_fix = alpha_model.compute_signals(rates_history, intraday_data,
                                            current_utc_time=pre_fix_time)["composite_skew_pips"]
alpha_at_fix = alpha_model.compute_signals(rates_history, intraday_data,
                                           current_utc_time=at_fix_time)["composite_skew_pips"]
alpha_off_fix = alpha_model.compute_signals(rates_history, intraday_data,
                                            current_utc_time=off_fix_time)["composite_skew_pips"]

prepared = wpm.prepare_features(subset)
if len(prepared) > 0:
    sample_rfq = prepared.iloc[[0]]
    q_pre = optimizer.optimal_quote(sample_rfq, current_utc_time=pre_fix_time,
                                    alpha_skew_pips=alpha_pre_fix)
    q_at = optimizer.optimal_quote(sample_rfq, current_utc_time=at_fix_time,
                                   alpha_skew_pips=alpha_at_fix)
    q_off = optimizer.optimal_quote(sample_rfq, current_utc_time=off_fix_time,
                                    alpha_skew_pips=alpha_off_fix)

    def _fmt_quote(label, q, alpha):
        return (f"    {label:<22} alpha={alpha:+.2f}p  "
                f"spread={q['optimal_spread_pips']:.2f}p  "
                f"P(hit)={q['hit_prob']*100:.1f}%  "
                f"E[α PnL]={q['expected_alpha_pnl_pips']:+.3f}p")

    print(f"  Direction-aware quoting (sample RFQ across three regimes):")
    print(_fmt_quote("Pre-fix (15:30 UTC):", q_pre, alpha_pre_fix))
    print(_fmt_quote("At-fix (16:01 UTC):", q_at, alpha_at_fix))
    print(_fmt_quote("Off-fix (10:00 UTC):", q_off, alpha_off_fix))
    print(f"    Pre-fix guardrails:  {q_pre.get('guardrails', [])}")
    print(f"    At-fix guardrails:   {q_at.get('guardrails', [])}")

# Per-RFQ alpha for the backtest: compute composite_skew once per unique
# (date, hour) and broadcast (cheap and captures intraday fix proximity).
def _alpha_lookup(timestamps_series, rates_history_df, intraday_df, alpha_mdl):
    ts = pd.to_datetime(timestamps_series)
    keys = list(zip(ts.dt.date, ts.dt.hour))
    cache = {}
    out = np.zeros(len(ts))
    for i, k in enumerate(keys):
        if k not in cache:
            t = datetime.combine(k[0], time(hour=k[1]))
            cache[k] = alpha_mdl.compute_signals(
                rates_history_df, intraday_df, current_utc_time=t
            )["composite_skew_pips"]
        out[i] = cache[k]
    return out

from datetime import time
alpha_per_rfq = _alpha_lookup(subset["timestamp"], rates_history, intraday_data, alpha_model)
print(f"\n  Per-RFQ alpha summary: mean={alpha_per_rfq.mean():+.3f}p  "
      f"std={alpha_per_rfq.std():.3f}p  "
      f"min={alpha_per_rfq.min():+.2f}p  max={alpha_per_rfq.max():+.2f}p  "
      f"|alpha|>0.5p: {int((np.abs(alpha_per_rfq) > 0.5).sum())}/{len(alpha_per_rfq)}")

# Smart optimizer: sees true alpha when picking spreads, and the world delivers it.
bt_smart = optimizer.backtest(subset,
                              alpha_skew_pips=alpha_per_rfq,
                              realized_alpha_pips="same")
# Alpha-blind optimizer: picks spreads under alpha=0, but the world still
# delivers true alpha. This is the apples-to-apples baseline for the lift.
bt_blind_under_truth = optimizer.backtest(subset,
                                          alpha_skew_pips=None,
                                          realized_alpha_pips=alpha_per_rfq)

# Decompose alpha-PnL into favorable-RFQ capture vs adverse-RFQ avoidance.
favorable_mask = bt_smart["opt_alpha_epnls_per_rfq"] > 0
favorable_capture = float(bt_smart["opt_alpha_epnls_per_rfq"][favorable_mask].sum())
adverse_avoidance = float((
    bt_blind_under_truth["opt_alpha_epnls_per_rfq"]
    - bt_smart["opt_alpha_epnls_per_rfq"]
)[~favorable_mask].sum())  # negative numbers -> we avoided them
n_favorable = int(favorable_mask.sum())
n_adverse = int((~favorable_mask & (bt_smart["opt_alpha_epnls_per_rfq"] < 0)).sum())

value_of_info = bt_smart["total_expected_pnl"] - bt_blind_under_truth["total_expected_pnl"]
realized_lift = bt_smart["total_realized_pnl"] - bt_blind_under_truth["total_realized_pnl"]

print(f"\n  Backtest Results ({len(subset)} RFQs):")
print(f"    Total E[PnL]:                 {bt_smart['total_expected_pnl']:.2f} pips")
print(f"    Total realized PnL:           {bt_smart['total_realized_pnl']:.2f} pips")
print(f"    Avg optimal spread:           {bt_smart['avg_spread']:.3f} pips")
print(f"    Avg P(hit):                   {bt_smart['avg_hit_prob']*100:.1f}%")
print(f"    Guardrails triggered:         {bt_smart['n_guardrails_triggered']}/{len(subset)}")
print(f"\n  Value of the W-shape signal:")
print(f"    Alpha-aware E[PnL]:           {bt_smart['total_expected_pnl']:.2f} pips")
print(f"    Alpha-blind E[PnL] under true drift: {bt_blind_under_truth['total_expected_pnl']:.2f} pips")
print(f"    Value-of-information lift:    {value_of_info:+.2f} pips E[PnL]; "
      f"{realized_lift:+.2f} pips realized")
print(f"    Favorable RFQs ({n_favorable}/{len(subset)}): captured  "
      f"{favorable_capture:+.2f} pips of edge")
print(f"    Adverse RFQs   ({n_adverse}/{len(subset)}): avoided   "
      f"{-adverse_avoidance:+.2f} pips of toxic flow")

comparison = optimizer.quote_vs_flat(subset, flat_spread_pips=1.0,
                                     alpha_skew_pips=alpha_per_rfq)
if isinstance(comparison, pd.DataFrame) and len(comparison) >= 3:
    opt_row = comparison.iloc[0]
    flat_row = comparison.iloc[1]
    imp_row = comparison.iloc[2]
    print(f"\n  Optimizer vs Flat 1.0 pip:")
    print(f"    Optimizer E[PnL]: {opt_row['total_expected_pnl']:.2f} pips, "
          f"Flat E[PnL]: {flat_row['total_expected_pnl']:.2f} pips")
    print(f"    Improvement: {imp_row['total_expected_pnl']:+.2f} pips "
          f"({imp_row['avg_hit_prob']:+.4f} hit rate)")


###############################################################################
# STEP 4: Hedge Sizing & Market Impact
###############################################################################

print("\n" + "=" * 70)
print("  STEP 4: Hedge Sizing & Market Impact")
print("  (Delta Hedge -> CME FX Futures, Almgren-Chriss Cost Model)")
print("=" * 70)

from module_c_execution.market_impact import AlmgrenChrissModel, FX_FUTURES

# --- 4a: Delta hedge sizing ---
print("\n-- 4a: Delta Hedge Sizing --")
print(f"\n  {'Swap':<30} {'Delta (USD)':>12} {'Futures':>8} {'Contracts':>10} {'Direction':>10}")
print("  " + "-" * 75)
execution_hedge_map = {}
for swap in portfolio:
    pricer = pricers[swap.pair]
    delta = pricer.delta(swap.notional, swap.maturity_years) * swap.sign()
    hedge = ra.best_futures_hedge(swap)
    dir_label = "BUY" if swap.direction == "buy_base" else "SELL"
    ccy = swap.pair.split("/")[0]
    print(f"  {dir_label} {swap.notional/1e6:.0f}MM {ccy} {swap.maturity_years:.2f}Y"
          f"{'':>12} {delta:>12,.0f} {hedge['ticker']:>8} {hedge['n_contracts']:>+10} {'BUY' if hedge['n_contracts'] > 0 else 'SELL':>10}")
    execution_hedge_map[swap.pair] = hedge

# --- 4b: Almgren-Chriss market impact ---
print("\n-- 4b: Market Impact (Almgren-Chriss) --")
impact_model = AlmgrenChrissModel()

print(f"\n  {'Ticker':<6} {'Contracts':>10} {'Perm (pips)':>12} {'Temp (pips)':>12} {'Cost (pips)':>12} {'Horizon':>10}")
print("  " + "-" * 65)

execution_costs_by_pair = {}
for swap in portfolio:
    hedge = execution_hedge_map[swap.pair]
    ticker = hedge["ticker"]
    n_contracts = max(abs(hedge["n_contracts"]), 1)
    imp = impact_model.cost_for_futures(ticker, n_contracts)
    execution_costs_by_pair[swap.pair] = imp.total_cost_pips
    print(f"  {ticker:<6} {n_contracts:>10} {imp.permanent_cost_pips:>12.4f} {imp.temporary_cost_pips:>12.4f} "
          f"{imp.total_cost_pips:>12.4f} {imp.optimal_horizon_minutes:>8.1f}m")

print("\n  Reference: 200 contracts per ticker")
print(f"  {'Ticker':<6} {'Contracts':>10} {'Perm (pips)':>12} {'Temp (pips)':>12} {'Cost (pips)':>12} {'Horizon':>10}")
print("  " + "-" * 65)
for ticker in FX_TICKERS:
    imp = impact_model.cost_for_futures(ticker, 200)
    print(f"  {ticker:<6} {200:>10} {imp.permanent_cost_pips:>12.4f} {imp.temporary_cost_pips:>12.4f} "
          f"{imp.total_cost_pips:>12.4f} {imp.optimal_horizon_minutes:>8.1f}m")


###############################################################################
# STEP 5: Execution
###############################################################################

print("\n" + "=" * 70)
print("  STEP 5: Execution")
print("  (TWAP/VWAP/Adaptive Strategies, L2 Slippage Backtest via KDB+)")
print("=" * 70)

from module_c_execution.execution_scheduler import compare_strategies
from module_c_execution.order_simulator import OrderSimulator

# --- 5a: Strategy comparison per hedge leg ---
print("\n-- 5a: Execution Strategy Comparison (per hedge leg from Step 4) --")
simulator = OrderSimulator()

for swap in portfolio:
    hedge = execution_hedge_map[swap.pair]
    ticker = hedge["ticker"]
    n_contracts = max(abs(hedge["n_contracts"]), 1)
    strategies = compare_strategies(n_contracts, kappa=1.5)
    dir_label = "BUY" if hedge["n_contracts"] > 0 else "SELL"
    ccy = swap.pair.split("/")[0]
    print(f"\n  {swap.pair} -> {dir_label} {n_contracts} {ticker}:")
    print(f"  {'Strategy':<12} {'First 3':>10} {'Last 3':>10} {'Max':>8}")
    print("  " + "-" * 42)
    for name, slices in strategies.items():
        first3 = sum(s.quantity for s in slices[:3])
        last3 = sum(s.quantity for s in slices[-3:])
        max_q = max(s.quantity for s in slices)
        print(f"  {name:<12} {first3:>10} {last3:>10} {max_q:>8}")

# --- 5b: Slippage backtest (L2 book-walking via KDB+) ---
print("\n-- 5b: Slippage Backtest (L2 via KDB+) --")

kdb = KDBInterface(host=KDB_HOST, port=KDB_PORT)
tick_count = kdb.tick_data_count()
print(f"  KDB+ tick_data: {tick_count} rows -- using L2 book-walking")
kdb.close()

print(f"\n  Per-hedge-leg L2 slippage (real book depth, 3 strategies):")
print(f"  {'Pair':<10} {'Ticker':<6} {'Contracts':>10} {'Strategy':<12} {'Slippage (pips)':>16}")
print("  " + "-" * 58)
# Store realised slippage per pair so Step 6 can deduct ACTUAL hedge cost
# from markout P&L, instead of the AC theoretical prediction from Step 4b.
# We use TWAP as the baseline strategy (defensible default; matches AC's
# constant-rate trading assumption).
realised_slippage_by_pair: dict[str, float] = {}
for swap in portfolio:
    hedge = execution_hedge_map[swap.pair]
    ticker = hedge["ticker"]
    n_contracts = max(abs(hedge["n_contracts"]), 1)
    bt = simulator.backtest_strategies(
        ticker=ticker, num_contracts=n_contracts, kappa=1.5,
        use_l2=True, kdb_host=KDB_HOST, kdb_port=KDB_PORT,
        date_str=TICK_DATA_DATE,
        start_time="09:30", end_time="16:00",
    )
    for strat_name, stats in bt.items():
        slippage = stats.get("slippage_pips", stats.get("slippage_bps", 0))
        print(f"  {swap.pair:<10} {ticker:<6} {n_contracts:>10} {strat_name:<12} {slippage:>15.4f}")
        if strat_name == "TWAP":
            realised_slippage_by_pair[swap.pair] = float(slippage)

# Predicted (AC model) vs Realised (TWAP book walk) hedge-cost comparison.
# This is the number that feeds Step 6's markout P&L decomposition.
print(f"\n  Predicted (Almgren-Chriss) vs Realised (TWAP L2 book walk):")
print(f"  {'Pair':<10} {'Ticker':<6} {'AC pred':>10} {'TWAP real':>12} {'Diff (pips)':>13}")
print("  " + "-" * 55)
for swap in portfolio:
    pair = swap.pair
    ac = float(execution_costs_by_pair.get(pair, 0.0))
    real = float(realised_slippage_by_pair.get(pair, 0.0))
    diff = real - ac
    ticker = execution_hedge_map[pair]["ticker"]
    print(f"  {pair:<10} {ticker:<6} {ac:>10.4f} {real:>12.4f} {diff:>+13.4f}")

# --- 5c: Impact curve + trajectory charts (6E hedge leg) ---
print("\n-- 5c: Execution Analysis Charts (6E) --")
eur_hedge = execution_hedge_map.get("EUR/USD", {})
eur_contracts = max(abs(eur_hedge.get("n_contracts", 80)), 1)
strategies = compare_strategies(eur_contracts, kappa=1.5)
eur_impact = impact_model.cost_for_futures("6E", eur_contracts)
curve_data = impact_model.impact_curve(eur_contracts, FX_FUTURES["6E"]["avg_daily_volume"],
                                        FX_FUTURES["6E"]["daily_vol_pips"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

horizon_key = "horizon_minutes" if "horizon_minutes" in curve_data else "horizons"
cost_key = "total_cost_pips" if "total_cost_pips" in curve_data else "total"
axes[0].plot(curve_data[horizon_key], curve_data[cost_key],
             "o-", color=JPM_COLORS["blue"], linewidth=2)
axes[0].axvline(x=eur_impact.optimal_horizon_minutes, color=JPM_COLORS["red"],
                linestyle="--", label=f"Optimal={eur_impact.optimal_horizon_minutes:.0f}min")
axes[0].set_xlabel("Execution Horizon (minutes)")
axes[0].set_ylabel("Total Cost (pips)")
axes[0].set_title(f"6E Market Impact Curve ({eur_contracts} contracts)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for name, slices in strategies.items():
    cum = [s.cumulative_quantity for s in slices]
    t = [s.time_fraction for s in slices]
    axes[1].plot(t, cum, "o-", label=name, linewidth=2, markersize=4)
axes[1].set_xlabel("Time Fraction")
axes[1].set_ylabel("Cumulative Contracts")
axes[1].set_title(f"Execution Trajectories ({eur_contracts} contracts)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "execution_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/execution_analysis.png")


###############################################################################
# STEP 6: P&L Analysis
###############################################################################

print("\n" + "=" * 70)
print("  STEP 6: P&L Analysis")
print("  (Fix-Conditioned Markouts, P&L Decomposition)")
print("=" * 70)

from module_b_trading.markout_pnl import FXMarkoutAnalyzer

# --- 6a: Markouts ---
print("\n-- 6a: Markout Analysis --")
filled = rfqs[rfqs["was_hit"] == True].copy()

# Prefer realised TWAP slippage (Step 5b L2 book walk) over the AC theoretical
# prediction (Step 4b). Fall back pair-by-pair to AC if the L2 backtest produced
# no value -- both dicts share the {pair: pips} schema FXMarkoutAnalyzer expects.
markout_costs = dict(execution_costs_by_pair)
markout_costs.update(realised_slippage_by_pair)
analyzer = FXMarkoutAnalyzer(execution_costs=markout_costs)

print("  Hedge-cost source per pair (used in markout P&L):")
for pair in G4_PAIRS:
    src = "L2 TWAP realised" if pair in realised_slippage_by_pair else "AC theoretical (fallback)"
    print(f"    {pair}: {markout_costs.get(pair, 0):.4f} pips  ({src})")

markouts = analyzer.compute_markouts(filled)
print(f"\n  Markout Summary ({len(markouts)} filled trades):")
print(f"    {'Horizon':<12} {'Mean (pips)':>12} {'Median':>10} {'Std':>10}")
print("    " + "-" * 48)
for col, label in [("markout_1m", "1-min"), ("markout_5m", "5-min"),
                    ("markout_30m", "30-min"), ("markout_1h", "1-hour"), ("markout_1d", "1-day")]:
    if col in markouts.columns:
        print(f"    {label:<12} {markouts[col].mean():>+11.3f}  {markouts[col].median():>+9.3f}  {markouts[col].std():>9.3f}")

# --- 6b: Fix-conditioned markouts (THE KEY RESULT) ---
print("\n-- 6b: Markouts by Fix Proximity (Key Result) --")
fix_markouts = analyzer.markout_by_fix_proximity(filled)
if isinstance(fix_markouts, pd.DataFrame) and len(fix_markouts) > 0:
    print(f"\n  {'Proximity':<12}", end="")
    for col in [c for c in fix_markouts.columns if "markout" in c.lower() or "mean" in c.lower()]:
        print(f" {col:>12}", end="")
    print()
    print("  " + "-" * 60)
    for _, row in fix_markouts.iterrows():
        prox = row.get("fix_proximity", row.get("proximity", row.index if hasattr(row, "index") else ""))
        print(f"    {str(prox):<12}", end="")
        for col in [c for c in fix_markouts.columns if "markout" in c.lower() or "mean" in c.lower()]:
            print(f" {row[col]:>+11.3f}", end="")
        print()

print("\n  Expected: Pre-fix fills show BETTER markouts (trading with the fix drift)")
print("  At-fix fills show WORSE markouts (adverse selection from fix order flow)")

# --- 6c: P&L decomposition ---
print("\n-- 6c: P&L Decomposition --")
pnl = analyzer.pnl_decomposition(filled)
report = analyzer.summary_report(pnl)

n_fills = report.get('n_fills', len(pnl))
print(f"\n  {n_fills} filled trades:")
print(f"    Mean P&L:        {report.get('total_pnl_mean', 0):+.3f} pips")
print(f"    Median P&L:      {report.get('total_pnl_median', 0):+.3f} pips")
print(f"    Std Dev:         {report.get('total_pnl_std', 0):.3f} pips")
print(f"    % Profitable:    {report.get('pct_profitable', 0) * 100:.1f}%")
print(f"    Sharpe (ann):    {report.get('sharpe_annualized', report.get('sharpe_ratio', 0)):.2f}")
# Show every component mean. The five named components sum to total_pnl
# exactly by construction; markout_drift captures whatever the realized
# post-fill mid move delivered beyond the modeled fix/carry contributions.
for comp in ["edge_pips", "fix_alpha_pips", "carry_pips", "hedge_cost_pips", "markout_drift_pips"]:
    if comp in pnl.columns:
        print(f"    Mean {comp:<22}: {pnl[comp].mean():+.3f} pips")
# Also surface the report dict's components if it had extra ones not above
components_dict = report.get('components', {})
if isinstance(components_dict, dict):
    for comp, stats in components_dict.items():
        if comp in {"edge_pips", "fix_alpha_pips", "carry_pips", "hedge_cost_pips",
                    "markout_drift_pips", "residual_pips"}:
            continue  # already printed
        val = stats.get('mean', 0) if isinstance(stats, dict) else stats
        print(f"    Mean {comp:<22}: {val:+.3f} pips")

# P&L summary chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# P&L decomposition: additive identity
#   total_pnl = edge + fix_alpha + carry + hedge_cost + markout_drift
# The "drift" component is the realized post-fill mid move that the four
# modeled components don't try to attribute. It is NOT model error.
components = ["edge_pips", "fix_alpha_pips", "carry_pips", "hedge_cost_pips", "markout_drift_pips"]
comp_labels = ["Edge\n(half-spread)", "Fix Alpha\n(W-shape)", "Carry\n(rate diff)",
               "Hedge Cost\n(L2 slippage)", "Markout Drift\n(post-fill mid move)"]
means = []
for comp in components:
    if comp in pnl.columns:
        means.append(pnl[comp].mean())
    elif comp == "markout_drift_pips" and "residual_pips" in pnl.columns:
        # Backwards compat: legacy name
        means.append(pnl["residual_pips"].mean())
    else:
        means.append(0)

total_pnl_mean = pnl.get("total_pnl_pips", pd.Series(means).sum()).mean() if hasattr(pnl, "get") else sum(means)
colors_pnl = [JPM_COLORS["green"] if v >= 0 else JPM_COLORS["red"] for v in means]
axes[0].bar(comp_labels, means, color=colors_pnl, alpha=0.8)
# Reference line: total mean P&L. The bars must sum to this by the identity.
axes[0].axhline(y=total_pnl_mean, color=JPM_COLORS["blue"], linewidth=1.5,
                linestyle="--", label=f"Total mean P&L: {total_pnl_mean:+.2f}p")
axes[0].set_ylabel("Mean P&L (pips)")
axes[0].set_title("P&L Decomposition by Component\n(bars sum to Total — drift is NOT residual error)")
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].axhline(y=0, color="black", linewidth=0.8)
axes[0].legend(loc="best", fontsize=9)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

# Markout by fix proximity
if isinstance(fix_markouts, pd.DataFrame) and len(fix_markouts) > 0:
    markout_col = [c for c in fix_markouts.columns if "markout_5m" in c.lower() or "mean" in c.lower()]
    if markout_col:
        prox_labels = fix_markouts.get("fix_proximity", fix_markouts.get("proximity", fix_markouts.index)).astype(str)
        vals = fix_markouts[markout_col[0]].values
        colors_fix = [JPM_COLORS["green"] if v >= 0 else JPM_COLORS["red"] for v in vals]
        axes[1].bar(range(len(vals)), vals, color=colors_fix, alpha=0.8, tick_label=prox_labels.values)
        axes[1].set_ylabel("Mean Markout (pips)")
        axes[1].set_title("5-min Markout by Fix Proximity")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].axhline(y=0, color="black", linewidth=0.8)

plt.tight_layout()
fig.savefig(output_dir / "pnl_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: output/pnl_analysis.png")


###############################################################################
# STEP 7: C++ Kernel Benchmark
###############################################################################

print("\n" + "=" * 70)
print("  STEP 7: C++ Kernel Benchmark")
print("  (Bootstrap, FX Forward, Adaptive Schedule, Almgren-Chriss)")
print("=" * 70)

import fx_pricing_kernel

print(f"  C++ fx_pricing_kernel loaded: {fx_pricing_kernel.__file__}")

# --- Helper: convert Python CurveInstrument -> C++ CurveInstrument ---
def _to_cpp_instrument(inst):
    if isinstance(inst, dict):
        itype = 0 if inst.get("type", "deposit") == "deposit" else 1
        mat = float(inst["maturity_years"])
        rate = float(inst["rate"])
        dc_map = {"ACT/360": 0, "ACT/365": 1, "30/360": 2}
        dc = dc_map.get(inst.get("day_count", "ACT/360"), 0)
        freq = int(inst.get("payment_frequency", 2))
    else:
        itype = 0 if getattr(inst, "type", "deposit") == "deposit" else 1
        mat = float(inst.maturity_years)
        rate = float(inst.rate)
        dc_map = {"ACT/360": 0, "ACT/365": 1, "30/360": 2}
        dc = dc_map.get(getattr(inst, "day_count", "ACT/360"), 0)
        freq = int(getattr(inst, "payment_frequency", 2))
    return fx_pricing_kernel.CurveInstrument(itype, mat, rate, dc, freq)

# --- 7a: Bootstrap benchmark (C++ vs Python) ---
print("\n-- 7a: Curve Bootstrap --")
n_iterations = 100

t0 = time.perf_counter()
for _ in range(n_iterations):
    cpp_insts = [_to_cpp_instrument(i) for i in usd_instruments]
    cpp_curve = fx_pricing_kernel.bootstrap_curve(cpp_insts)
t_cpp = (time.perf_counter() - t0) / n_iterations * 1000

t0 = time.perf_counter()
for _ in range(n_iterations):
    _ = bs.bootstrap(usd_instruments, VALUATION_DATE)
t_py = (time.perf_counter() - t0) / n_iterations * 1000

print(f"  Bootstrap ({n_iterations} iterations):")
print(f"    Python:  {t_py:.2f} ms/call")
print(f"    C++:     {t_cpp:.2f} ms/call")
if t_cpp > 0:
    print(f"    Speedup: {t_py / t_cpp:.1f}x")

# Validate C++ curve matches Python
print(f"\n  Validation: C++ vs Python discount factors:")
print(f"    {'Tenor':<8} {'Python D(t)':>12} {'C++ D(t)':>12} {'Diff':>10}")
for t in [1, 2, 5, 10, 30]:
    d_py = usd_curve.df(t)
    d_cpp = cpp_curve.df(t)
    diff = abs(d_py - d_cpp)
    print(f"    {t}Y{'':<4} {d_py:>12.6f} {d_cpp:>12.6f} {diff:>10.2e}")

# --- 7b: FX Forward Pricer benchmark ---
print("\n-- 7b: FX Forward Pricer --")
# Build foreign curve in C++
eur_rate = 0.0290
eur_instruments_cpp = [
    fx_pricing_kernel.CurveInstrument(0, 1/360, eur_rate, 0, 1),
    fx_pricing_kernel.CurveInstrument(1, 1.0, eur_rate, 0, 1),
    fx_pricing_kernel.CurveInstrument(1, 5.0, eur_rate + 0.003, 0, 1),
]
cpp_eur_curve = fx_pricing_kernel.bootstrap_curve(eur_instruments_cpp)
cpp_fx_pricer = fx_pricing_kernel.FXForwardPricer(cpp_curve, cpp_eur_curve, fx_spots["EUR/USD"])

n_fwd = 10_000
t0 = time.perf_counter()
for _ in range(n_fwd):
    fwd_val = cpp_fx_pricer.forward(1.0)
t_cpp_fwd = (time.perf_counter() - t0) / n_fwd * 1e6
fwd_pts = cpp_fx_pricer.forward_points(1.0)
print(f"  FX Forward ({n_fwd} iterations): {t_cpp_fwd:.1f} us/call")
print(f"  EUR/USD 1Y forward: {fwd_val:.4f}, points: {fwd_pts:.4f}")

# --- 7c: Adaptive Schedule benchmark ---
print("\n-- 7c: Adaptive Schedule --")
n_bench = 10_000
from module_c_execution.execution_scheduler import AdaptiveScheduler as PyAdaptive

t0 = time.perf_counter()
for _ in range(n_bench):
    PyAdaptive(kappa=1.5, n_slices=13).schedule(1000)
t_py_sched = (time.perf_counter() - t0) / n_bench * 1e6

t0 = time.perf_counter()
for _ in range(n_bench):
    fx_pricing_kernel.adaptive_schedule(1000, 1.5, 13)
t_cpp_sched = (time.perf_counter() - t0) / n_bench * 1e6

print(f"  Adaptive Schedule ({n_bench} iterations, 1000 contracts):")
print(f"    Python:  {t_py_sched:.1f} us/call")
print(f"    C++:     {t_cpp_sched:.1f} us/call")
if t_cpp_sched > 0:
    print(f"    Speedup: {t_py_sched / t_cpp_sched:.1f}x")

# --- 7d: Almgren-Chriss benchmark ---
print("\n-- 7d: Almgren-Chriss Impact --")
t0 = time.perf_counter()
for _ in range(n_bench):
    fx_pricing_kernel.AlmgrenChrissEngine().total_cost(200, 250000, 55.0, 30.0)
t_cpp_impact = (time.perf_counter() - t0) / n_bench * 1e6
print(f"  Almgren-Chriss ({n_bench} iterations): {t_cpp_impact:.1f} us/call")

# --- Summary table ---
print(f"\n  {'Operation':<35} {'Python':>10} {'C++':>10} {'Speedup':>10}")
print("  " + "-" * 68)
print(f"  {'Curve bootstrap (' + str(len(usd_instruments)) + ' nodes)':<35} {t_py:>8.2f}ms {t_cpp:>8.2f}ms {t_py/max(t_cpp,0.001):>9.1f}x")
print(f"  {'Adaptive schedule (1000 ctrs)':<35} {t_py_sched:>7.1f} us {t_cpp_sched:>7.1f} us {t_py_sched/max(t_cpp_sched,0.001):>9.1f}x")
print(f"  {'FX forward price':<35} {'--':>10} {t_cpp_fwd:>7.1f} us {'C++ only':>10}")
print(f"  {'Almgren-Chriss impact':<35} {'--':>10} {t_cpp_impact:>7.1f} us {'C++ only':>10}")


###############################################################################
# DONE
###############################################################################

print("\n" + "=" * 70)
print("  DEMO COMPLETE")
print("=" * 70)
print("All steps executed successfully.")
print(f"Plots saved to {output_dir}/")
print("  - interpolation_comparison.png  (USD OIS forward rates)")
print("  - fx_delta_ladder.png           (portfolio spot deltas)")
print("  - w_shape_pattern.png           (Krohn et al. W-shaped fix pattern)")
print("  - execution_analysis.png        (impact curve + trajectories)")
print("  - pnl_analysis.png              (P&L decomposition + fix markouts)")

# Clean up KDB+ if we started it
if kdb_proc is not None:
    print(f"\nStopping KDB+ (PID {kdb_proc.pid})...")
    kdb_proc.terminate()
    kdb_proc.wait(timeout=5)
    print("KDB+ stopped.")
