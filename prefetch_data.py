"""Prefetch all real Databento data for the FX fix-pattern project.

This script orchestrates the four-component data fetch the project relies on:

  1. MBP-1 (top-of-book) fix-pattern data: every business day in the
     ``WIN_START -> VAL_DATE`` window, 4 FX continuous contracts, 15-17 UTC.
  2. MBP-10 (deep book) data: 10 selected execution-simulation days,
     4 FX continuous contracts, 13-17 UTC.
  3. SOFR (SR3) OHLCV-1d strip: 8 continuous contracts, ~252 trading days
     ending at the valuation date (used for USD OIS bootstrap).
  4. Back-month CME FX OHLCV-1d curve: 16 contracts on the valuation date
     (used for the carry / cost-of-carry block).

All four components are quoted authoritatively via
``databento.Historical().metadata.get_cost(...)`` BEFORE any fetch happens.
The script refuses to run if the quoted total exceeds ``--budget`` (default
$30). Real cost on the canonical 2026-01-28 -> 2026-04-28 window has been
~$5.28.

Usage
-----
    # Quote-only, no spend:
    python prefetch_data.py --dry-run

    # Quote, prompt, then fetch on confirmation:
    python prefetch_data.py

    # Quote and fetch non-interactively:
    python prefetch_data.py --yes

    # Override window (still quote-then-confirm):
    python prefetch_data.py --val-date 2026-05-30 --win-start 2026-03-01 --yes

Cache & idempotency
-------------------
Every component is cached as a parquet under ``data/databento_cache/``. If
a cache file already exists the corresponding fetch is skipped (the script
never re-spends on data already on disk). Empty Databento responses are
recorded as zero-byte marker files so subsequent runs also skip them.
After the per-component fetches complete, the aggregate 15-min returns
parquet at ``data/cme_15m_cache.parquet`` is rebuilt via
``module_b_trading.fix_alpha_signals.build_cme_15m_cache``.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "databento_cache"
CME_15M_CACHE = PROJECT_ROOT / "data" / "cme_15m_cache.parquet"

VAL_DATE = "2026-04-28"
WIN_START = "2026-01-28"
FX_TICKERS = ["6E", "6B", "6J", "6A"]
EXEC_SIM_DAYS = [
    "2026-02-06", "2026-03-05", "2026-03-06", "2026-03-18", "2026-03-19",
    "2026-03-31", "2026-04-03", "2026-04-16", "2026-02-20", "2026-04-22",
]
DEFAULT_BUDGET_USD = 30.0
DATASET = "GLBX.MDP3"        # CME (USD: SR3, SR1; FX futures)
DATASET_ICE = "IFLL.IMPACT"  # ICE Futures Europe (GBP: SO3, SOA)
DATASET_EUREX = "XEUR.EOBI"  # Eurex (EUR: FST3, FEMP)

# 1y EOD windows for the foreign-currency RFR strips.  Mirror the SR3 layout
# so each parquet covers ~252 trading days ending on the valuation date.
FCY_RFR_STRIPS = [
    # (cache_key_prefix, label, dataset, root, n_contracts)
    ("sofr_sr1",   "SR1 1M SOFR",     DATASET,       "SR1",  7),
    ("sonia_so3",  "SO3 3M SONIA",    DATASET_ICE,   "SO3",  8),
    ("sonia_soa",  "SOA 1M SONIA",    DATASET_ICE,   "SOA",  6),
    ("estr_fst3",  "FST3 3M ESTR",    DATASET_EUREX, "FST3", 8),
    ("estr_femp",  "FEMP ECB ESTR",   DATASET_EUREX, "FEMP", 6),
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _client():
    """Build a databento.Historical client. Falls back to .env for the API key."""
    import databento as db
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DATABENTO_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        return None
    return db.Historical(api_key)


# ---------------------------------------------------------------------------
# Quote helpers
# ---------------------------------------------------------------------------

def _quote_one(client, label: str, **kwargs) -> float:
    """Return the metadata.get_cost dollar amount for a single request, or 0.0 on error."""
    try:
        cost = client.metadata.get_cost(**kwargs)
        if isinstance(cost, dict):
            usd = float(cost.get("cost", cost.get("usd", 0.0)))
        else:
            usd = float(cost)
        return usd
    except Exception as exc:
        print(f"  [quote-error] {label}: {exc}")
        return 0.0


def _build_request_plan(val_date: str, win_start: str) -> list:
    """Materialise the list of (label, kwargs) requests to quote and fetch.

    Returns a list of dicts with keys: ``label``, ``cache_key``, ``kwargs``.
    Any caller passes the same plan to both the quote loop and the fetch loop
    so the two stay in sync.
    """
    fx_cont = [f"{t}.c.0" for t in FX_TICKERS]
    plan = []

    # 1. MBP-1 fix-pattern, per business day
    bdays = pd.bdate_range(win_start, val_date)
    for d in bdays:
        ds = d.strftime("%Y-%m-%d")
        plan.append({
            "label": f"MBP-1 {ds}",
            "cache_key": f"mbp1_fx_{ds}",
            "kwargs": dict(
                dataset=DATASET, symbols=fx_cont, schema="mbp-1",
                stype_in="continuous",
                start=f"{ds}T15:00:00", end=f"{ds}T17:00:00",
            ),
        })

    # 2. MBP-10 deep book on selected days
    for ds in EXEC_SIM_DAYS:
        plan.append({
            "label": f"MBP-10 {ds}",
            "cache_key": f"mbp10_fx_{ds}",
            "kwargs": dict(
                dataset=DATASET, symbols=fx_cont, schema="mbp-10",
                stype_in="continuous",
                start=f"{ds}T13:00:00", end=f"{ds}T17:00:00",
            ),
        })

    # 3. SOFR strip: 1y window ending at val_date
    sofr_start = (pd.Timestamp(val_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    plan.append({
        "label": "SOFR SR3.c.0..7 OHLCV-1d (1y)",
        "cache_key": f"sofr_sr3_strip_1y_to_{val_date}",
        "kwargs": dict(
            dataset=DATASET,
            symbols=[f"SR3.c.{i}" for i in range(8)],
            schema="ohlcv-1d", stype_in="continuous",
            start=f"{sofr_start}T00:00:00", end=f"{val_date}T23:59:59",
        ),
    })

    # 3b. Foreign-currency / extra USD RFR strips: SR1, SO3, SOA, FST3, FEMP.
    # All share the same 1y EOD window ending at val_date; differ only by
    # dataset, root and contract count.
    for cache_prefix, label, dataset, root, n in FCY_RFR_STRIPS:
        plan.append({
            "label": f"{label} {root}.c.0..{n - 1} OHLCV-1d (1y)",
            "cache_key": f"{cache_prefix}_strip_1y_to_{val_date}",
            "kwargs": dict(
                dataset=dataset,
                symbols=[f"{root}.c.{i}" for i in range(n)],
                schema="ohlcv-1d", stype_in="continuous",
                start=f"{sofr_start}T00:00:00", end=f"{val_date}T23:59:59",
            ),
        })

    # 4a. Back-month FX curve — primary: actual daily close (last trade).
    # Parent symbology returns all active contracts; the loader filters spreads
    # and non-quarterly month codes via `_parse_fx_symbol`.
    #
    # Why not `.c.N` continuous: Databento's FX continuous map rolls through
    # monthlies too (K/N/Q/V/X), so e.g. `6E.c.0` on 2026-04-28 returned the
    # May (K) contract instead of Jun (M) — produced a 100+ bp CIP-vs-CME
    # mispricing because the wrong expiry was associated with each price.
    plan.append({
        "label": "FX back-month curve (parent ohlcv-1d, last trade)",
        "cache_key": f"fx_back_month_curve_{val_date}",
        "kwargs": dict(
            dataset=DATASET,
            symbols=[f"{t}.FUT" for t in FX_TICKERS],
            schema="ohlcv-1d", stype_in="parent",
            start=f"{val_date}T00:00:00", end=f"{val_date}T23:59:59",
        ),
    })

    # 4b. Back-month FX settles — fallback: CME's official daily settlement
    # (statistics stat_type==3) for every active contract, even ones that did
    # not trade. The loader prefers the ohlcv-1d close above; this fills in
    # only the contracts ohlcv-1d misses (e.g. illiquid long-tenor IMMs like
    # 6BM7 that never traded on the val_date).
    plan.append({
        "label": "FX back-month settles (parent statistics, CME official)",
        "cache_key": f"fx_back_month_settles_{val_date}",
        "kwargs": dict(
            dataset=DATASET,
            symbols=[f"{t}.FUT" for t in FX_TICKERS],
            schema="statistics", stype_in="parent",
            start=f"{val_date}T00:00:00", end=f"{val_date}T23:59:59",
        ),
    })

    return plan


def quote_total(client, plan: Optional[list] = None) -> tuple:
    """Quote every planned request and return (total_usd, line_items).

    ``line_items`` is a list of ``(label, cost_usd)`` tuples in plan order.
    """
    if plan is None:
        plan = _build_request_plan(VAL_DATE, WIN_START)

    line_items = []
    total = 0.0
    for req in plan:
        usd = _quote_one(client, req["label"], **req["kwargs"])
        line_items.append((req["label"], usd))
        total += usd
    return total, line_items


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_one(client, label: str, cache_path: Path, **kwargs) -> bool:
    """Idempotent fetch: skip if cache_path exists, else fetch and write parquet."""
    if cache_path.exists():
        print(f"  [skip] {label} (cached -> {cache_path.name})")
        return True
    try:
        data = client.timeseries.get_range(**kwargs)
        df = data.to_df()
        if df.empty:
            print(f"  [empty] {label} -> writing zero-byte marker")
            cache_path.write_text("")
            return True
        df.reset_index().to_parquet(cache_path, index=False)
        print(f"  [ok] {label} -> {cache_path.name} ({len(df)} rows)")
        return True
    except Exception as exc:
        print(f"  [fail] {label} -> {exc}")
        return False


def fetch_all(client, plan: Optional[list] = None) -> dict:
    """Run every fetch in the plan. Returns counts by component family."""
    if plan is None:
        plan = _build_request_plan(VAL_DATE, WIN_START)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    counts = {
        "mbp1_days": 0, "mbp10_days": 0,
        "sofr_sr3": 0, "sofr_sr1": 0,
        "sonia_so3": 0, "sonia_soa": 0,
        "estr_fst3": 0, "estr_femp": 0,
        "fx_curve": 0,
    }
    for req in plan:
        cache_path = CACHE_DIR / f"{req['cache_key']}.parquet"
        ok = _fetch_one(client, req["label"], cache_path, **req["kwargs"])
        if not ok:
            continue
        key = req["cache_key"]
        if key.startswith("mbp1_fx_"):
            counts["mbp1_days"] += 1
        elif key.startswith("mbp10_fx_"):
            counts["mbp10_days"] += 1
        elif key.startswith("sofr_sr3_"):
            counts["sofr_sr3"] += 1
        elif key.startswith("sofr_sr1_"):
            counts["sofr_sr1"] += 1
        elif key.startswith("sonia_so3_"):
            counts["sonia_so3"] += 1
        elif key.startswith("sonia_soa_"):
            counts["sonia_soa"] += 1
        elif key.startswith("estr_fst3_"):
            counts["estr_fst3"] += 1
        elif key.startswith("estr_femp_"):
            counts["estr_femp"] += 1
        elif key.startswith("fx_back_month_curve_"):
            counts["fx_curve"] += 1
    return counts


# ---------------------------------------------------------------------------
# Post-fetch aggregation
# ---------------------------------------------------------------------------

def build_cme_15m_cache_after_fetch() -> int:
    """Refresh data/cme_15m_cache.parquet from MBP-1 parquets in the cache dir."""
    try:
        from module_b_trading.fix_alpha_signals import build_cme_15m_cache
    except Exception as exc:
        print(f"  [warn] could not import build_cme_15m_cache: {exc}")
        return 0
    try:
        return int(build_cme_15m_cache(CACHE_DIR, CME_15M_CACHE))
    except Exception as exc:
        print(f"  [warn] build_cme_15m_cache failed: {exc}")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Prefetch real Databento data for the FX fix-pattern project."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Quote only; never fetch. Exits 0 after printing the quote.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive confirmation prompt.")
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                   help=f"Hard cap in USD (default: {DEFAULT_BUDGET_USD}).")
    p.add_argument("--val-date", type=str, default=None,
                   help="Override valuation date (YYYY-MM-DD).")
    p.add_argument("--win-start", type=str, default=None,
                   help="Override fix-pattern window start (YYYY-MM-DD).")
    return p.parse_args()


def _print_line_items(line_items: list, total: float) -> None:
    print("=" * 78)
    print("DATABENTO COST QUOTE  (metadata.get_cost, no billing)")
    print("=" * 78)
    for label, cost in line_items:
        print(f"  {label:<60} ${cost:>10.4f}")
    print("-" * 78)
    print(f"  {'GRAND TOTAL':<60} ${total:>10.4f}")
    print("=" * 78)


def main() -> int:
    global VAL_DATE, WIN_START

    args = _parse_args()
    if args.val_date:
        VAL_DATE = args.val_date
    if args.win_start:
        WIN_START = args.win_start

    client = _client()
    if client is None:
        print("ERROR: DATABENTO_API_KEY not found in env or .env", file=sys.stderr)
        return 2

    plan = _build_request_plan(VAL_DATE, WIN_START)

    print(f"Valuation date:      {VAL_DATE}")
    print(f"Fix-pattern window:  {WIN_START} -> {VAL_DATE}")
    print(f"Planned requests:    {len(plan)}")
    print()

    total, line_items = quote_total(client, plan)
    _print_line_items(line_items, total)

    if total > args.budget:
        print(f"ERROR: quoted total ${total:.4f} exceeds budget ${args.budget:.2f}",
              file=sys.stderr)
        return 2

    if args.dry_run:
        print("Dry run -- no fetch performed")
        return 0

    if not args.yes:
        try:
            answer = input(f"Proceed with ${total:.4f} fetch? [y/N]: ").strip()
        except EOFError:
            answer = ""
        if answer.lower() != "y":
            print("Aborted by user")
            return 0

    print()
    print("=" * 78)
    print("FETCHING")
    print("=" * 78)
    counts = fetch_all(client, plan)

    print()
    print("=" * 78)
    print("BUILDING 15-MIN AGGREGATE CACHE")
    print("=" * 78)
    n_rows = build_cme_15m_cache_after_fetch()
    print(f"  cme_15m_cache.parquet -> {n_rows} rows")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  MBP-1 days fetched/cached:   {counts['mbp1_days']}")
    print(f"  MBP-10 days fetched/cached:  {counts['mbp10_days']}")
    print(f"  SOFR SR3 strip:              {counts['sofr_sr3']}")
    print(f"  SOFR SR1 strip:              {counts['sofr_sr1']}")
    print(f"  SONIA SO3 strip:             {counts['sonia_so3']}")
    print(f"  SONIA SOA strip:             {counts['sonia_soa']}")
    print(f"  ESTR FST3 strip:             {counts['estr_fst3']}")
    print(f"  ESTR FEMP strip:             {counts['estr_femp']}")
    print(f"  FX back-month curve:         {counts['fx_curve']}")
    print(f"  15-min aggregate rows:       {n_rows}")
    print(f"  Quoted total:                ${total:.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
