"""
Module B: FX Trading - Core Market-Making Engine
=================================================

Implements the W-shaped FX fix pattern (Krohn, Mueller & Whelan, Journal of Finance 2024)
for market making across G4 FX pairs (EUR/USD, GBP/USD, JPY/USD, AUD/USD).

Submodules:
    fx_pricer         - CIP-based FX forward/swap pricing and Greeks
    risk_analytics    - Bump-and-revalue risk, VaR, CME futures hedge mapping
    fix_alpha_signals - W-shaped fix pattern alpha and composite signal model
    rfq_generator     - Synthetic FX RFQ flow generator
    win_probability   - Logistic win probability model for FX RFQs
    quote_optimizer   - Fix-aware quote optimization with guardrails
    markout_pnl       - Fix-conditioned markout P&L analysis and decomposition
"""

from .fx_pricer import FXPricer, FXSwapSpec
from .risk_analytics import FXRiskAnalytics
from .fix_alpha_signals import FixAlphaModel, CompositeAlphaModel
from .rfq_generator import FXRFQGenerator
from .win_probability import FXWinProbabilityModel
from .quote_optimizer import FXQuoteOptimizer
from .markout_pnl import FXMarkoutAnalyzer

__all__ = [
    "FXPricer",
    "FXSwapSpec",
    "FXRiskAnalytics",
    "FixAlphaModel",
    "CompositeAlphaModel",
    "FXRFQGenerator",
    "FXWinProbabilityModel",
    "FXQuoteOptimizer",
    "FXMarkoutAnalyzer",
]
