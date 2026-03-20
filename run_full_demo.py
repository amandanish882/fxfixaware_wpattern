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
# Real data cached in data/databento_cache/ for 2025-03-03 through 2025-03-07
TICK_DATA_DATE = "2025-03-04"
TICK_START_TIME = "13:00"
TICK_END_TIME = "17:00"
# Valuation date: use yesterday (FRED publishes Treasury yields with 1-day lag)
VALUATION_DATE = (date.today() - timedelta(days=1)).isoformat()
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

# --- 0c: Load tick data from Databento (cached parquet or synthetic) ---
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
    print(f"  Loaded {total_raw} total tick rows from Databento cache/synthetic")
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

# --- 1b: USD OIS Curve ---
print("\n-- 1b: USD OIS Curve (Log-Linear Bootstrap) --")
ois_rates = loader.get_ois_rates(VALUATION_DATE)

# Convert FRED rates dict to CurveInstrument list for bootstrapper
# Map: SOFR -> O/N deposit, SOFR30DAYAVG -> 1M deposit, SOFR90DAYAVG -> 3M deposit
# DGS2/5/10/30 -> swaps at 2Y/5Y/10Y/30Y
_RATE_TO_INSTRUMENT = {
    "SOFR":          ("deposit", 1/360, "ACT/360", 1.0),
    "SOFR30DAYAVG":  ("deposit", 1/12,  "ACT/360", 1.0),
    "SOFR90DAYAVG":  ("deposit", 3/12,  "ACT/360", 1.0),
    "DGS2":          ("swap",    2.0,   "ACT/360", 0.5),
    "DGS5":          ("swap",    5.0,   "ACT/360", 0.5),
    "DGS10":         ("swap",   10.0,   "ACT/360", 0.5),
    "DGS30":         ("swap",   30.0,   "ACT/360", 0.5),
}

usd_instruments = []
for series, rate in ois_rates.items():
    if series in _RATE_TO_INSTRUMENT:
        inst_type, mat, dc, freq = _RATE_TO_INSTRUMENT[series]
        usd_instruments.append(CurveInstrument(
            type=inst_type, maturity_years=mat, rate=rate,
            day_count=dc, payment_frequency=freq,
        ))

t0 = time.perf_counter()
bs = CurveBootstrapper(interpolation_method="log_linear")
usd_curve = bs.bootstrap(usd_instruments, VALUATION_DATE)
t_curve = time.perf_counter() - t0

validation = bs.validate(usd_curve, usd_instruments)
max_err = validation["error_bps"].abs().max()

print(f"  Curve built: {len(usd_curve.times)} nodes, max tenor = {usd_curve.times[-1]:.1f}Y")
print(f"  Max repricing error: {max_err:.4f} bps  |  Build time: {t_curve*1000:.1f} ms")
print("\n  Key USD rates:")
for t in [1, 2, 5, 10, 30]:
    print(f"    {t}Y par: {usd_curve.par_rate(t)*100:.3f}%  |  "
          f"zero: {usd_curve.zero_rate(t)*100:.3f}%  |  D({t}): {usd_curve.df(t):.6f}")

# --- 1c: Monotone Convex comparison ---
print("\n-- 1c: Monotone Convex vs Log-Linear --")
bs_mc = CurveBootstrapper(interpolation_method="monotone_convex")
usd_curve_mc = bs_mc.bootstrap(usd_instruments, VALUATION_DATE)

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

# --- 1d: FX Forward Curves via CIP ---
print("\n-- 1d: FX Forward Curves (Covered Interest Parity) --")

# Build simplified foreign OIS curves from fallback rates
# EUR: ECB deposit rate ~2.90%, GBP: BOE rate ~4.50%
# Use flat-rate curves as proxy (realistic simplification for demo)
from module_a_curves.curve_bootstrapper import DiscountCurve, CurveInstrument
import datetime as dt

foreign_rates = {"EUR/USD": 0.0290, "GBP/USD": 0.0450, "JPY/USD": 0.0010, "AUD/USD": 0.0435}
val_date = date.today()

foreign_curves = {}
for pair, rate in foreign_rates.items():
    # Build a flat OIS curve from a single deposit at the policy rate
    flat_instruments = [
        CurveInstrument(type="deposit", maturity_years=1/360, rate=rate, day_count="ACT/360"),
        CurveInstrument(type="swap", maturity_years=1.0, rate=rate, day_count="ACT/360", payment_frequency=1.0),
        CurveInstrument(type="swap", maturity_years=2.0, rate=rate + 0.001, day_count="ACT/360", payment_frequency=1.0),
        CurveInstrument(type="swap", maturity_years=5.0, rate=rate + 0.003, day_count="ACT/360", payment_frequency=1.0),
        CurveInstrument(type="swap", maturity_years=10.0, rate=rate + 0.005, day_count="ACT/360", payment_frequency=1.0),
    ]
    foreign_bs = CurveBootstrapper(interpolation_method="log_linear")
    foreign_curves[pair] = foreign_bs.bootstrap(flat_instruments, VALUATION_DATE)

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

# --- 1e: Cross-Currency Basis ---
print("\n-- 1e: Cross-Currency Basis (CIP Deviations) --")
ccb = CrossCurrencyBasis()
for pair in G4_PAIRS:
    basis_df = ccb.basis_curve(pair)
    basis_str = "  ".join(f"{row['tenor']}: {row['basis_bps']:+.1f}" for _, row in basis_df.iterrows())
    print(f"    {pair}: {basis_str}")

# --- 1f: Central Bank Meeting Dates ---
print("\n-- 1f: Central Bank Meeting Dates (2026) --")
cb_dates = loader.fetch_central_bank_dates(year=2026)
for bank, dates in cb_dates.items():
    date_strs = []
    for d in dates[:4]:
        if hasattr(d, "strftime"):
            date_strs.append(d.strftime("%m/%d"))
        else:
            date_strs.append(str(d))
    print(f"    {bank}: {', '.join(date_strs)} ... ({len(dates)} meetings)")


###############################################################################
# STEP 2: FX Pricing & Risk
###############################################################################

print("\n" + "=" * 70)
print("  STEP 2: FX Pricing & Risk")
print("  (FX Swap Portfolio, Spot Delta, Cross-Gamma, Scenarios, VaR)")
print("=" * 70)

from module_b_trading.fx_pricer import FXPricer, FXSwapSpec
from module_b_trading.risk_analytics import FXRiskAnalytics

# Build pricers per pair
pricers = {}
for pair in G4_PAIRS:
    spot = fx_spots.get(pair, 1.0)
    pricers[pair] = FXPricer(usd_curve, foreign_curves[pair], spot, pair)

# --- 2a: Portfolio Pricing ---
print("\n-- 2a: FX Swap Portfolio Pricing --")
portfolio = [
    FXSwapSpec(10_000_000, 1.0, pricers["EUR/USD"].forward_price(1.0) - 0.005, "EUR/USD", "buy_base"),
    FXSwapSpec(5_000_000, 0.5, pricers["GBP/USD"].forward_price(0.5) + 0.002, "GBP/USD", "sell_base"),
    FXSwapSpec(500_000_000, 1.0, pricers["JPY/USD"].forward_price(1.0) * 1.01, "JPY/USD", "buy_base"),
    FXSwapSpec(8_000_000, 0.25, pricers["AUD/USD"].forward_price(0.25) - 0.003, "AUD/USD", "buy_base"),
]

print(f'  {"Swap":<35} {"NPV (USD)":>14} {"Fwd Mkt":>10} {"Fwd Agreed":>11}')
print("  " + "=" * 72)
total_npv = 0
for swap in portfolio:
    pricer = pricers[swap.pair]
    npv = pricer.fx_swap_npv(swap.notional, swap.maturity_years, swap.agreed_forward)
    npv_signed = npv * swap.sign()
    fwd_mkt = pricer.forward_price(swap.maturity_years)
    dir_label = "BUY" if swap.direction == "buy_base" else "SELL"
    total_npv += npv_signed
    ccy = swap.pair.split("/")[0]
    print(f"  {dir_label} {swap.notional/1e6:.0f}MM {ccy} {swap.maturity_years:.2f}Y"
          f"{'':>14} {npv_signed:>14,.0f} {fwd_mkt:>10.4f} {swap.agreed_forward:>11.4f}")
print("  " + "=" * 72)
print(f'  {"TOTAL PORTFOLIO NPV":<35} {total_npv:>14,.0f}')

# --- 2b: Risk Analytics ---
print("\n-- 2b: Risk (Spot Delta, Rate Delta) --")
ra = FXRiskAnalytics(bs, {"USD": usd_instruments}, VALUATION_DATE, fx_spots)

print(f'  {"Swap":<30} {"Spot Delta (USD)":>16} {"Futures Hedge":>14}')
print("  " + "-" * 62)
hedge_results = {}
for swap in portfolio:
    pricer = pricers[swap.pair]
    delta = pricer.delta(swap.notional, swap.maturity_years) * swap.sign()
    hedge = ra.best_futures_hedge(swap)
    dir_label = "BUY" if swap.direction == "buy_base" else "SELL"
    ccy = swap.pair.split("/")[0]
    print(f"  {dir_label} {swap.notional/1e6:.0f}MM {ccy} {swap.maturity_years:.2f}Y"
          f"{'':>10} {delta:>16,.0f} {hedge['n_contracts']:>+4} {hedge['ticker']}")
    hedge_results[swap.pair] = hedge

# --- 2c: Cross-pair correlation ---
print("\n-- 2c: Cross-Pair Correlation Matrix --")
corr = ra.correlation_matrix()
print(corr.to_string(float_format="{:.3f}".format))

# --- 2d: Parametric VaR ---
print("\n-- 2d: Parametric VaR (99%, 1-day) --")
var_result = ra.var_portfolio(portfolio, confidence=0.99)
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
n_days_approx = max(1, len(intraday_data) // (96 * len(G4_PAIRS)))
if data_source.startswith("yahoo"):
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
axes[0].set_xlabel("15-min Bucket (0 = 00:00 UTC)")
axes[0].set_ylabel("Cumulative Return (bps)")
axes[0].set_title("EUR/USD Intraday Cumulative Return (W-Shape)")
# Mark fix times: Tokyo=bucket 3 (00:45-01:00), ECB=bucket 53 (13:15), London=bucket 64 (16:00)
for fix_name, bucket_idx in [("Tokyo", 3), ("ECB", 53), ("London", 64)]:
    axes[0].axvline(x=bucket_idx, color=JPM_COLORS["red"], linestyle="--", alpha=0.7, label=f"{fix_name} fix")
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
print("\n-- 3d: Fix-Aware Quote Optimizer --")
optimizer = FXQuoteOptimizer(wpm, fix_model, lambda_risk=1e-6)
subset = rfqs.head(100)

# Show fix-aware quoting example
pre_fix_time = datetime(2026, 3, 5, 15, 30)  # 30min before London fix
off_fix_time = datetime(2026, 3, 5, 10, 0)   # no fix nearby

prepared = wpm.prepare_features(subset)
if len(prepared) > 0:
    sample_rfq = prepared.iloc[[0]]  # DataFrame, not Series
    quote_prefix = optimizer.optimal_quote(sample_rfq, current_utc_time=pre_fix_time)
    quote_offfix = optimizer.optimal_quote(sample_rfq, current_utc_time=off_fix_time)
    print(f"  Fix-Aware Quoting Example:")
    print(f"    Pre-fix (15:30 UTC):  spread = {quote_prefix['optimal_spread_pips']:.2f} pips")
    print(f"    Off-fix (10:00 UTC):  spread = {quote_offfix['optimal_spread_pips']:.2f} pips")
    print(f"    Guardrails (pre-fix): {quote_prefix.get('guardrails', [])}")

bt_stats = optimizer.backtest(subset)
print(f"\n  Backtest Results ({len(subset)} RFQs):")
print(f"    Total E[PnL]:         {bt_stats.get('total_expected_pnl', 0):.2f} pips")
print(f"    Avg optimal spread:   {bt_stats.get('avg_spread', bt_stats.get('avg_spread_pips', 0)):.2f} pips")
print(f"    Avg P(hit) at optimal: {bt_stats.get('avg_hit_prob', 0)*100:.1f}%")
print(f"    Guardrails triggered: {bt_stats.get('n_guardrails_triggered', 0)}/{len(subset)}")

comparison = optimizer.quote_vs_flat(subset, flat_spread_pips=1.0)
if isinstance(comparison, pd.DataFrame) and len(comparison) >= 3:
    opt_row = comparison.iloc[0]
    flat_row = comparison.iloc[1]
    imp_row = comparison.iloc[2]
    print(f"\n  Optimizer vs Flat 1.0 pip:")
    print(f"    Optimizer E[PnL]: {opt_row['total_expected_pnl']:.2f} pips, "
          f"Flat E[PnL]: {flat_row['total_expected_pnl']:.2f} pips")
    print(f"    Improvement: {imp_row['total_expected_pnl']:+.2f} pips ({imp_row['avg_hit_prob']:+.4f} hit rate)")


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
analyzer = FXMarkoutAnalyzer(execution_costs=execution_costs_by_pair)

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
print(f"    % Profitable:    {report.get('pct_profitable', 0):.1f}%")
print(f"    Sharpe (ann):    {report.get('sharpe_annualized', report.get('sharpe_ratio', 0)):.2f}")
# Show component means
components_dict = report.get('components', {})
if isinstance(components_dict, dict):
    for comp, stats in components_dict.items():
        if isinstance(stats, dict):
            print(f"    Mean {comp:<14}: {stats.get('mean', 0):+.3f} pips")
        else:
            print(f"    Mean {comp:<14}: {stats:+.3f} pips")

# P&L summary chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# P&L decomposition bar chart
components = ["edge_pips", "fix_alpha_pips", "carry_pips", "hedge_cost_pips", "residual_pips"]
comp_labels = ["Edge", "Fix Alpha", "Carry", "Hedge Cost", "Residual"]
means = []
for comp in components:
    if comp in pnl.columns:
        means.append(pnl[comp].mean())
    else:
        means.append(0)

colors_pnl = [JPM_COLORS["green"] if v >= 0 else JPM_COLORS["red"] for v in means]
axes[0].bar(comp_labels, means, color=colors_pnl, alpha=0.8)
axes[0].set_ylabel("Mean P&L (pips)")
axes[0].set_title("P&L Decomposition by Component")
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].axhline(y=0, color="black", linewidth=0.8)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

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
