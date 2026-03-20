"""
Module C — Execution analytics for CME FX futures.

Submodules
----------
market_impact
    Almgren-Chriss model calibrated for 6E, 6B, 6J, 6A.
execution_scheduler
    TWAP, VWAP, and Adaptive (sinh-urgency) scheduling.
order_simulator
    L2 book-walking simulator and strategy backtester.
"""

from .market_impact import AlmgrenChrissModel, FX_FUTURES, ImpactResult
from .execution_scheduler import (
    AdaptiveScheduler,
    ExecutionSlice,
    TWAPScheduler,
    VWAPScheduler,
    compare_strategies,
)
from .order_simulator import OrderSimulator

__all__ = [
    "AlmgrenChrissModel",
    "FX_FUTURES",
    "ImpactResult",
    "AdaptiveScheduler",
    "ExecutionSlice",
    "TWAPScheduler",
    "VWAPScheduler",
    "compare_strategies",
    "OrderSimulator",
]
