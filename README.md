<p align="center">
  <img src="output/w_shape_pattern.png" alt="W-Shaped FX Fix Pattern" width="750"/>
</p>

<h1 align="center">FX Fix-Aware Market Making</h1>

<p align="center">
  <em>Exploiting the W-shaped intraday pattern around global FX fixes &mdash; Tokyo, ECB, and London WMR &mdash; for systematic G4 currency market-making on CME FX futures.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white" alt="C++17"/>
  <img src="https://img.shields.io/badge/pybind11-2.11-orange?logo=python&logoColor=white" alt="pybind11"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/KDB+-tick%20store-00A3E0?logo=kx&logoColor=white" alt="KDB+"/>
  <img src="https://img.shields.io/badge/Databento-CME%20MBP--10-6C3483" alt="Databento"/>
</p>

---

## Overview

This project implements the **W-shaped FX fix pattern** documented in [Krohn, Mueller & Whelan (2024)](#references): the USD systematically appreciates in the 30--60 minutes before each of the three daily FX fixes (Tokyo 00:55 UTC, ECB 13:15 UTC, London WMR 16:00 UTC) and depreciates after, producing a characteristic W-shape in cumulative intraday returns.

The system builds a **fix-aware market-making engine** that combines fix alpha signals with carry, momentum, and mean-reversion in a composite quoting model, prices via covered interest parity (CIP), hedges through CME FX futures, and tracks P&L through fix-conditioned markout analysis.

### Key Capabilities

| Capability | Detail |
|---|---|
| **Curve construction** | 8-node USD OIS + EUR/GBP OIS proxy curves, monotone convex interpolation, max repricing error 2.21 bps |
| **FX forward pricing** | CIP-based: $F(T) = S \cdot D_f(T) / D_d(T)$, cross-currency basis modelling |
| **Fix alpha signals** | W-shape pattern, carry, momentum, mean-reversion; composite alpha (50/25/15/10 weights) |
| **Quote optimization** | $E[\text{PnL}](s) = P(\text{hit} \mid s) \cdot (s - \text{cost}) - \lambda \delta^2 + \text{fix\_adj}$; win model AUC 0.783 |
| **Execution** | Almgren-Chriss impact model, TWAP/VWAP/Adaptive scheduling on CME FX futures |
| **P&L attribution** | Fix-conditioned markouts: pre-fix +0.92 pips, at-fix -0.10 pips; Sharpe 0.96 |
| **C++ kernel** | `curve_engine.h`, `fx_pricer.h`, `execution_engine.h` via pybind11 |
| **Tick data** | Databento CME MBP-10 (6E, 6B, 6J, 6A), KDB+ tick store on port 5001 |

---

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set API keys
export FRED_API_KEY="your_key"
export DATABENTO_API_KEY="your_key"

# 3. (Optional) Start KDB+ tick store
q tick.q -p 5001

# 4. Build C++ kernel
cd shared/cpp_kernel && mkdir build && cd build && cmake .. && make && cd ../../..

# 5. Run the full pipeline
python run_full_demo.py

# 6. Run tests (46 tests across 4 files)
python -m pytest module_a_curves/tests/ module_b_trading/tests/ module_c_execution/tests/ -v
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_full_demo.py                             │
│                     (8-step pipeline orchestrator)                   │
├────────────────┬──────────────────────┬─────────────────────────────┤
│  Module A      │  Module B            │  Module C                   │
│  Curves        │  Trading             │  Execution                  │
├────────────────┼──────────────────────┼─────────────────────────────┤
│ data_loader    │ fix_alpha_signals    │ market_impact               │
│ interpolation  │ win_probability      │ execution_scheduler         │
│ curve_boot-    │ quote_optimizer      │ order_simulator             │
│   strapper     │ rfq_generator        │                             │
│ fx_forward_    │ fx_pricer            │                             │
│   curve        │ risk_analytics       │                             │
│                │ markout_pnl          │                             │
├────────────────┴──────────────────────┴─────────────────────────────┤
│                       shared/                                       │
│  plot_style · date_utils · databento_loader · kdb_interface         │
├─────────────────────────────────────────────────────────────────────┤
│                  shared/cpp_kernel/                                  │
│  curve_engine.h · fx_pricer.h · execution_engine.h · pybind11      │
├─────────────────────────────────────────────────────────────────────┤
│                      Data Layer                                     │
│  FRED API (FX spots + OIS rates) · Databento (CME MBP-10) · KDB+  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Walkthrough

### Step 0 &mdash; Infrastructure &amp; Data Ingestion

Loads CME FX futures tick data (MBP-10) via Databento for the four G4 contracts and seeds the KDB+ tick store.

```
[Step 0] Infrastructure & Data Ingestion
  4000 tick snapshots loaded (1000 per ticker)
  Tickers: 6E (EUR/USD, 125K EUR), 6B (GBP/USD, 62.5K GBP),
           6J (JPY/USD, 12.5M JPY), 6A (AUD/USD, 100K AUD)
```

| CME Ticker | Pair | Contract Size | Tick Size | Tick Value |
|---|---|---|---|---|
| **6E** | EUR/USD | 125,000 EUR | 0.00005 | $6.25 |
| **6B** | GBP/USD | 62,500 GBP | 0.0001 | $6.25 |
| **6J** | JPY/USD | 12,500,000 JPY | 0.0000005 | $6.25 |
| **6A** | AUD/USD | 100,000 AUD | 0.0001 | $10.00 |

---

### Step 1 &mdash; FX Curve Construction

Bootstraps an 8-node USD OIS discount curve using monotone convex interpolation ([Hagan & West, 2006](#references)), then constructs FX forward curves via covered interest parity (CIP):

$$F(T) = S \cdot \frac{D_f(T)}{D_d(T)}$$

where $S$ is the spot rate, $D_f(T)$ is the foreign discount factor, and $D_d(T)$ is the domestic (USD) discount factor. Cross-currency basis (CIP deviations) are modelled explicitly: EUR basis ranges from -15 to -25 bps, JPY from -30 to -50 bps.

```
[Step 1] FX Curve Construction
  8-node curve, max repricing error: 2.21 bps, build time: 0.4 ms
  USD par rates: 2Y 4.157%, 5Y 4.336%, 10Y 4.624%, 30Y 4.756%
  EUR/USD 1Y forward: 1.0968, forward points: +135.6 pips
```

<p align="center">
  <img src="output/interpolation_comparison.png" alt="Interpolation Comparison" width="700"/>
</p>

*Left: USD OIS forward rates under log-linear vs. monotone convex interpolation. Right: difference between the two methods, showing the oscillations that log-linear introduces.*

---

### Step 2 &mdash; FX Pricing &amp; Risk

Computes portfolio-level NPV, spot delta per currency pair, cross-gamma, and parametric VaR (99%, 1-day):

$$\text{VaR}_{99\%} = \sum_i \delta_i \cdot \sigma_i \cdot z_{0.99} \cdot S_i$$

where $\delta_i$ is the spot delta for pair $i$, $\sigma_i$ is daily volatility, and $z_{0.99} = 2.326$.

```
[Step 2] FX Pricing & Risk
  Portfolio NPV: $48,227
  VaR (99%, 1-day): $134,804
```

<p align="center">
  <img src="output/fx_delta_ladder.png" alt="FX Delta Ladder" width="600"/>
</p>

*Spot delta by currency pair. EUR/USD and AUD/USD are long USD; GBP/USD is short USD.*

---

### Step 3 &mdash; Fix Alpha Signals &amp; Quote Optimization

Implements the W-shaped fix pattern from Krohn, Mueller & Whelan (2024). The USD appreciates before each of the three daily fixes and depreciates after:

| Fix | Window | Return (bps) | t-stat |
|---|---|---|---|
| **London WMR** | [-60, -30] min | -2.22 | -3.32 |
| **London WMR** | [-30, -15] min | -1.46 | -3.23 |

The composite alpha signal combines four components:

$$\alpha_{\text{composite}} = 0.50 \cdot \alpha_{\text{fix}} + 0.25 \cdot \alpha_{\text{carry}} + 0.15 \cdot \alpha_{\text{mom}} + 0.10 \cdot \alpha_{\text{MR}}$$

A logistic regression win-probability model (AUC = 0.783, accuracy = 71.5%) feeds into the quote optimizer:

$$E[\text{PnL}](s) = P(\text{hit} \mid s) \cdot (s - \text{cost}) - \lambda \delta^2 + \text{fix\_adj}$$

```
[Step 3] Fix Alpha Signals & Quote Optimization
  W-shape London fix: [-60,-30] = -2.22 bps (t=-3.32)
                      [-30,-15] = -1.46 bps (t=-3.23)
  Win model AUC: 0.783, Accuracy: 71.5%
  Optimizer E[PnL]: 63.31 pips, improvement: +14.46 vs flat 1.0 pip
```

---

### Step 4 &mdash; Hedge Sizing &amp; Market Impact

Converts spot delta into CME FX futures hedge orders and estimates market impact using the Almgren-Chriss framework:

$$\text{Impact}(Q) = \sigma \cdot \left(\frac{Q}{V}\right)^{0.6}$$

where $Q$ is the order size in contracts, $V$ is average daily volume, and $\sigma$ is daily volatility in pips.

```
[Step 4] Hedge Sizing & Market Impact
  6E: 71 contracts -> impact: 0.61 pips
  6J: 5994 contracts -> impact: 78.02 pips
```

---

### Step 5 &mdash; Execution Simulation

Simulates execution across three scheduling algorithms (TWAP, VWAP, Adaptive) using L2 order-book data from KDB+:

```
[Step 5] Execution Simulation
  6E TWAP slippage: 0.37 pips
  6B Adaptive slippage: 0.67 pips
```

<p align="center">
  <img src="output/execution_analysis.png" alt="Execution Analysis" width="700"/>
</p>

*Left: 6E market impact curve (total cost vs. execution horizon). The red dot marks the optimal 1-minute horizon. Right: cumulative execution trajectories for TWAP, VWAP, and Adaptive schedulers.*

---

### Step 6 &mdash; P&amp;L Attribution &amp; Markout Analysis

Decomposes P&L by component (edge, fix alpha, carry, hedge cost, residual) and computes fix-conditioned markouts &mdash; the expected 5-minute return conditional on proximity to a fix window:

| Regime | Markout (pips) | Interpretation |
|---|---|---|
| **pre_fix** | +0.92 | Informed flow; USD appreciation |
| **at_fix** | -0.10 | Adverse selection at fix |
| **post_fix** | +0.93 | Mean-reversion after fix |
| **neutral** | +0.55 | Baseline market-making edge |

```
[Step 6] P&L Attribution
  Mean P&L: +0.195 pips, Sharpe: 0.96
  Edge: +0.631 pips, Fix alpha: +0.090 pips
```

<p align="center">
  <img src="output/pnl_analysis.png" alt="P&L Analysis" width="700"/>
</p>

*Left: P&L decomposition by component. Right: 5-minute markouts conditioned on fix proximity &mdash; pre-fix and post-fix regimes are profitable; at-fix shows adverse selection.*

---

### Step 7 &mdash; C++ Kernel Benchmark

Benchmarks the pybind11 C++ kernel for latency-critical operations:

```
[Step 7] C++ Kernel Benchmark
  Bootstrap: 0.14 ms/iter
  Adaptive schedule: 153.8 us/call
```

| Kernel | Header | Function |
|---|---|---|
| **curve_engine.h** | Monotone convex bootstrap | Discount factors, forward rates |
| **fx_pricer.h** | CIP forward pricing | FX forwards, basis adjustment |
| **execution_engine.h** | Adaptive scheduler | Order slicing, impact estimation |

---

## Data Sources

| Source | Data | Detail |
|---|---|---|
| **FRED API** | FX spot rates | DEXUSEU, DEXUSUK, DEXJPUS, DEXUSAL |
| **FRED API** | OIS / policy rates | SOFR, ECBESTRVOLWGTTRMDMNRT (EUR), IUDSOIA (GBP) |
| **FRED API** | Treasury par rates | DGS2, DGS5, DGS10, DGS30 |
| **Databento** | CME FX futures | MBP-10 (6E, 6B, 6J, 6A) |
| **KDB+** | Tick storage | Port 5001, partitioned by date |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+, C++17 |
| **Curve engine** | NumPy, SciPy (monotone convex) |
| **ML** | scikit-learn (logistic regression, AUC), XGBoost |
| **Data** | pandas, pyarrow, fredapi, databento |
| **Tick store** | KDB+ / q (via qpython) |
| **C++ binding** | pybind11, CMake |
| **Visualization** | matplotlib, seaborn |

---

## Tests

46 tests across 4 test files:

| File | Tests | Coverage |
|---|---|---|
| `module_a_curves/tests/test_curves.py` | 12 | Bootstrapping, interpolation, forward pricing, CIP |
| `module_b_trading/tests/test_fx_trading.py` | 15 | Fix signals, win model, quote optimizer, markouts |
| `module_c_execution/tests/test_execution.py` | 11 | Impact model, TWAP/VWAP/Adaptive, slippage |
| `module_c_execution/tests/test_cpp_kernel.py` | 8 | C++ kernel parity vs Python, latency bounds |

```bash
python -m pytest module_a_curves/tests/ module_b_trading/tests/ module_c_execution/tests/ -v
```

---

## Project Structure

```
fx project/
├── run_full_demo.py              # 8-step pipeline orchestrator
├── config.py                     # Central configuration (API keys, FX constants, fix times)
├── requirements.txt
├── module_a_curves/
│   ├── data_loader.py            # FRED API data fetching (FX spots + OIS rates)
│   ├── interpolation.py          # Monotone convex interpolation (Hagan & West)
│   ├── curve_bootstrapper.py     # USD OIS discount curve bootstrap
│   ├── fx_forward_curve.py       # CIP-based FX forward pricing + basis
│   └── tests/
│       └── test_curves.py        # 12 tests
├── module_b_trading/
│   ├── fix_alpha_signals.py      # W-shape pattern (Krohn, Mueller & Whelan)
│   ├── win_probability.py        # Logistic regression win model (AUC 0.783)
│   ├── quote_optimizer.py        # E[PnL] quote optimization with fix adjustment
│   ├── rfq_generator.py          # Synthetic RFQ stream generator
│   ├── fx_pricer.py              # FX spot/forward pricer
│   ├── risk_analytics.py         # Delta, gamma, VaR, scenario analysis
│   ├── markout_pnl.py            # Fix-conditioned markout decomposition
│   └── tests/
│       └── test_fx_trading.py    # 15 tests
├── module_c_execution/
│   ├── market_impact.py          # Almgren-Chriss impact model
│   ├── execution_scheduler.py    # TWAP / VWAP / Adaptive scheduling
│   ├── order_simulator.py        # L2 order-book execution simulator
│   └── tests/
│       ├── test_execution.py     # 11 tests
│       └── test_cpp_kernel.py    # 8 tests
├── shared/
│   ├── plot_style.py             # JPM-style plot formatting
│   ├── date_utils.py             # Business day and tenor utilities
│   ├── databento_loader.py       # Databento CME FX futures loader
│   ├── kdb_interface.py          # KDB+ tick store read/write (port 5001)
│   └── cpp_kernel/
│       ├── include/
│       │   ├── curve_engine.h    # C++ monotone convex bootstrap
│       │   ├── fx_pricer.h       # C++ CIP forward pricer
│       │   └── execution_engine.h # C++ adaptive scheduler
│       ├── bindings/
│       │   └── pybind_module.cpp # pybind11 Python bindings
│       ├── CMakeLists.txt
│       └── setup.py
└── output/
    ├── w_shape_pattern.png       # EUR/USD intraday W-shape around fixes
    ├── interpolation_comparison.png  # Log-linear vs monotone convex
    ├── fx_delta_ladder.png       # Portfolio spot delta by pair
    ├── execution_analysis.png    # Impact curve + execution trajectories
    └── pnl_analysis.png          # P&L decomposition + fix markouts
```

---

## References

1. **Krohn, I., Mueller, P., & Whelan, P.** (2024). Understanding the Global FX Fix. *Journal of Finance*. &mdash; Documents the W-shaped intraday pattern around the three daily FX fixes; the core paper behind this project.

2. **Barzykin, A., Bergault, P., & Gu&eacute;ant, O.** (2021&ndash;2023). Multi-Currency Optimal Market Making. HSBC framework for joint quoting across correlated FX pairs with inventory constraints.

3. **Cartea, &Aacute;., & S&aacute;nchez-Betancourt, L.** (2023). Toxic Flow Detection in FX Markets. Identifies adverse-selection regimes around fix windows using trade-flow imbalance signals.

4. **Almgren, R., & Chriss, N.** (2001). Optimal Execution of Portfolio Transactions. *Journal of Risk*, 3(2), 5&ndash;39. Square-root impact model for execution cost estimation.

5. **Cartea, &Aacute;., Jaimungal, S., & Penalva, J.** (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press. Theoretical foundations for market-making and optimal execution.

6. **Avellaneda, M., & Stoikov, S.** (2008). High-Frequency Trading in a Limit Order Book. *Quantitative Finance*, 8(3), 217&ndash;224. Inventory-penalized quoting framework.

7. **Hagan, P. S., & West, G.** (2006). Interpolation Methods for Curve Construction. *Applied Mathematical Finance*, 13(2), 89&ndash;129. Monotone convex interpolation for arbitrage-free yield curves.

8. **Bank for International Settlements** (2025). Triennial Central Bank Survey of Foreign Exchange and OTC Derivatives Markets. Global FX turnover: $9.6 trillion/day.
