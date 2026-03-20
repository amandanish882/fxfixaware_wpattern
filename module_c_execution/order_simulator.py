"""
L2 order-book walking simulator and strategy backtester for CME FX futures.

Simulates filling block orders by walking through Level-2 book depth,
and backtests TWAP/VWAP/Adaptive strategies against either live KDB+
snapshots or synthetic book data with realistic FX depth characteristics.

Synthetic books model:
- L1 spread: 0.5-1 tick (tight for liquid FX futures)
- L2-L5: spread widens ~0.5 tick per level
- Size: exponential decay from L1 (largest) to L10 (smallest)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .execution_scheduler import (
    AdaptiveScheduler,
    ExecutionSlice,
    TWAPScheduler,
    VWAPScheduler,
)
from .market_impact import FX_FUTURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic book generation
# ---------------------------------------------------------------------------

def _generate_synthetic_book(
    ticker: str,
    n_levels: int = 10,
    mid_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a realistic synthetic L2 order book snapshot.

    Models CME FX futures book depth with:
    - **L1 spread**: 0.5-1.0 ticks (e.g. 0.5 * 0.00005 = 0.0000025 for 6E)
    - **Level spacing**: ~0.5 tick per additional level
    - **Size decay**: exponential from L1 (e.g. 50 contracts) to L10 (e.g. 5)

    Parameters
    ----------
    ticker : str
        CME futures ticker (``"6E"``, ``"6B"``, ``"6J"``, ``"6A"``).
    n_levels : int
        Number of price levels on each side.  Default 10.
    mid_price : float, optional
        Mid-price to centre the book around.  If ``None``, uses a
        representative price: 1.0850 for 6E, 1.2650 for 6B, etc.

    Returns
    -------
    dict
        Keys: ``mid_price``, ``bids`` (list of [price, size] from best to
        worst), ``asks`` (list of [price, size] from best to worst),
        ``ticker``, ``tick_size``.

    Examples
    --------
    >>> book = _generate_synthetic_book("6E")
    >>> book["asks"][0]  # best ask: [1.08502500, 48]
    >>> book["bids"][0]  # best bid: [1.08497500, 52]
    """
    spec = FX_FUTURES[ticker]
    tick = spec["tick_size"]

    # Representative mid-prices for each contract
    default_mids = {
        "6E": 1.0850,
        "6B": 1.2650,
        "6J": 0.006700,
        "6A": 0.6550,
    }
    mid = mid_price if mid_price is not None else default_mids.get(ticker, 1.0000)

    # Half-spread: 0.5-1.0 ticks
    half_spread_ticks = random.uniform(0.5, 1.0)
    half_spread = half_spread_ticks * tick

    # Base L1 size: roughly proportional to daily volume
    base_size = max(10, spec["avg_daily_volume"] // 5000)  # e.g. 250000/5000 = 50

    bids: List[List[float]] = []
    asks: List[List[float]] = []

    for level in range(n_levels):
        # Spread widens by ~0.5 tick per level beyond L1
        level_offset = level * 0.5 * tick

        ask_price = mid + half_spread + level_offset
        bid_price = mid - half_spread - level_offset

        # Size decays exponentially: L1 gets base_size, L10 gets ~10% of that
        # decay = exp(-0.25 * level) gives ratio ~0.08 at level 9
        size_multiplier = np.exp(-0.25 * level)
        level_size = max(1, int(base_size * size_multiplier * random.uniform(0.8, 1.2)))

        asks.append([round(ask_price, 10), level_size])
        bids.append([round(bid_price, 10), level_size])

    return {
        "mid_price": mid,
        "bids": bids,
        "asks": asks,
        "ticker": ticker,
        "tick_size": tick,
    }


def _fetch_kdb_snapshots(
    ticker: str,
    n_snapshots: int,
    kdb_host: str,
    kdb_port: int,
    date_str: str,
    start_time: str,
    end_time: str,
) -> Optional[List[Dict[str, Any]]]:
    """Attempt to fetch L2 book snapshots from a KDB+ server.

    Connects via qPython (if available) and queries for evenly spaced
    book snapshots between *start_time* and *end_time*.

    Parameters
    ----------
    ticker : str
        CME futures ticker.
    n_snapshots : int
        Number of snapshots to retrieve.
    kdb_host : str
        KDB+ server hostname.
    kdb_port : int
        KDB+ server port.
    date_str : str
        Date string in ``"YYYY-MM-DD"`` format.
    start_time : str
        Start time in ``"HH:MM"`` format (e.g. ``"09:30"``).
    end_time : str
        End time in ``"HH:MM"`` format (e.g. ``"16:00"``).

    Returns
    -------
    list[dict] or None
        List of book snapshot dicts, or ``None`` if KDB+ is unreachable.
    """
    try:
        from qpython import qconnection  # type: ignore[import-untyped]

        q = qconnection.QConnection(host=kdb_host, port=kdb_port)
        q.open()

        # Query evenly spaced snapshots
        query = (
            f"select from book where ticker=`{ticker}, "
            f"date={date_str.replace('-', '.')}, "
            f"time within ({start_time.replace(':', '')}; {end_time.replace(':', '')})"
        )
        result = q(query)
        q.close()

        if result is None or len(result) == 0:
            logger.warning("KDB+ returned no data for %s on %s", ticker, date_str)
            return None

        # Sample n_snapshots evenly from the result
        indices = np.linspace(0, len(result) - 1, n_snapshots, dtype=int)
        snapshots = []
        for idx in indices:
            row = result[idx]
            snapshots.append({
                "mid_price": float(row.get("mid", 1.0)),
                "bids": row.get("bids", []),
                "asks": row.get("asks", []),
                "ticker": ticker,
                "tick_size": FX_FUTURES[ticker]["tick_size"],
            })

        return snapshots

    except ImportError:
        logger.info("qPython not installed — falling back to synthetic books.")
        return None
    except Exception as exc:
        logger.warning("KDB+ connection failed (%s) — falling back to synthetic books.", exc)
        return None


# ---------------------------------------------------------------------------
# Order simulator
# ---------------------------------------------------------------------------
class OrderSimulator:
    """Simulate order execution by walking L2 order-book depth.

    Provides book-walking fills and full strategy backtesting with either
    real KDB+ data or synthetic book fallback.

    Examples
    --------
    >>> sim = OrderSimulator()
    >>> book = _generate_synthetic_book("6E")
    >>> result = sim.walk_book(book, 80, side="buy")
    >>> result["avg_fill_price"]   # e.g. 1.0850375
    >>> result["slippage_pips"]    # e.g. 0.35
    >>> result["levels_consumed"]  # e.g. 3
    """

    def __init__(self) -> None:
        pass

    # -----------------------------------------------------------------
    # Book walking
    # -----------------------------------------------------------------
    def walk_book(
        self,
        book_snapshot: Dict[str, Any],
        quantity: int,
        side: str = "buy",
    ) -> Dict[str, Any]:
        """Walk through L2 book levels to fill a given quantity.

        For a **buy** order, we consume ask levels starting from the best
        (lowest) ask.  For a **sell**, we consume bid levels from the best
        (highest) bid.

        Parameters
        ----------
        book_snapshot : dict
            L2 book with keys ``mid_price``, ``bids``, ``asks``, ``tick_size``.
        quantity : int
            Number of contracts to fill.
        side : str
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        dict
            - ``avg_fill_price``: volume-weighted average fill price.
            - ``slippage_pips``: deviation from mid in pips (always >= 0).
            - ``levels_consumed``: how many L2 levels were touched.
            - ``total_cost``: ``avg_fill_price * quantity`` (notional proxy).

        Examples
        --------
        >>> sim = OrderSimulator()
        >>> book = _generate_synthetic_book("6E", mid_price=1.08500)
        >>> r = sim.walk_book(book, 30, "buy")
        >>> # Fills mostly at L1 ask ≈ 1.08503, slippage ≈ 0.3 pips
        """
        mid = book_snapshot["mid_price"]
        tick = book_snapshot.get("tick_size", 0.00005)

        # Select the appropriate side of the book
        if side == "buy":
            levels = book_snapshot["asks"]  # sorted best (lowest) first
        else:
            levels = book_snapshot["bids"]  # sorted best (highest) first

        remaining = float(quantity)
        filled_notional = 0.0
        filled_qty = 0.0
        levels_consumed = 0

        for price, size in levels:
            if remaining <= 0:
                break

            levels_consumed += 1
            fill_at_level = min(remaining, float(size))
            filled_notional += fill_at_level * price
            filled_qty += fill_at_level
            remaining -= fill_at_level

        # If we exhausted all levels and still have remaining, fill at worst
        # price + 1 tick (market sweep scenario)
        if remaining > 0 and levels:
            worst_price = levels[-1][0]
            sweep_price = worst_price + tick if side == "buy" else worst_price - tick
            filled_notional += remaining * sweep_price
            filled_qty += remaining
            levels_consumed += 1

        avg_fill = filled_notional / filled_qty if filled_qty > 0 else mid

        # Slippage in pips (1 pip = 0.0001 for most pairs, 0.01 for JPY)
        # We derive pip size from tick_size: for 6J tick=0.0000005 → pip=0.000001
        # For others tick=0.00005 or 0.0001 → pip=0.0001
        pip_size = self._pip_size_from_tick(tick)
        raw_slippage = abs(avg_fill - mid) / pip_size

        return {
            "avg_fill_price": round(avg_fill, 10),
            "slippage_pips": round(raw_slippage, 4),
            "levels_consumed": levels_consumed,
            "total_cost": round(avg_fill * quantity, 6),
        }

    @staticmethod
    def _pip_size_from_tick(tick_size: float) -> float:
        """Derive pip size from tick size.

        For most FX pairs 1 pip = 0.0001.  For JPY crosses where tick is
        0.0000005, 1 pip = 0.000001 (since JPY quotes use 6+ decimals in
        the CME convention for 6J = JPY/USD).

        Parameters
        ----------
        tick_size : float
            The contract's tick size.

        Returns
        -------
        float
            One pip in price terms.

        Examples
        --------
        >>> OrderSimulator._pip_size_from_tick(0.00005)   # 6E → 0.0001
        0.0001
        >>> OrderSimulator._pip_size_from_tick(0.0000005) # 6J → 0.000001
        1e-06
        """
        if tick_size < 0.00001:
            # JPY-type contract (6J): pip = 0.000001
            return 0.000001
        else:
            # Standard FX: pip = 0.0001
            return 0.0001

    # -----------------------------------------------------------------
    # Strategy backtest
    # -----------------------------------------------------------------
    def backtest_strategies(
        self,
        ticker: str,
        num_contracts: int,
        kappa: float = 1.5,
        use_l2: bool = True,
        kdb_host: str = "localhost",
        kdb_port: int = 5001,
        date_str: str = "2026-03-05",
        start_time: str = "09:30",
        end_time: str = "16:00",
    ) -> Dict[str, Dict[str, float]]:
        """Backtest TWAP, VWAP, and Adaptive strategies against L2 book data.

        For each strategy:
        1. Generate the execution schedule (13 slices).
        2. Obtain 13 L2 book snapshots — either from KDB+ or synthetic.
        3. Walk the book for each slice's quantity.
        4. Aggregate total slippage and cost.

        Parameters
        ----------
        ticker : str
            CME futures ticker (e.g. ``"6E"``).
        num_contracts : int
            Total contracts to execute (e.g. 200).
        kappa : float
            Urgency for AdaptiveScheduler.  Default 1.5.
        use_l2 : bool
            If ``True``, attempt KDB+ first.  Default ``True``.
        kdb_host : str
            KDB+ hostname.  Default ``"localhost"``.
        kdb_port : int
            KDB+ port.  Default ``5001``.
        date_str : str
            Date for historical backtest.  Default ``"2026-03-05"``.
        start_time : str
            Session start in ``"HH:MM"``.  Default ``"09:30"``.
        end_time : str
            Session end in ``"HH:MM"``.  Default ``"16:00"``.

        Returns
        -------
        dict[str, dict[str, float]]
            Outer key: strategy name (``"TWAP"``, ``"VWAP"``, ``"Adaptive"``).
            Inner keys: ``avg_fill``, ``slippage_pips``, ``total_cost``.

        Examples
        --------
        >>> sim = OrderSimulator()
        >>> results = sim.backtest_strategies("6E", 100)
        >>> results["TWAP"]["slippage_pips"]   # e.g. 0.42
        >>> results["Adaptive"]["slippage_pips"]  # e.g. 0.38 (front-loaded)
        """
        if ticker not in FX_FUTURES:
            raise KeyError(
                f"Unknown ticker '{ticker}'. Available: {list(FX_FUTURES.keys())}"
            )

        spec = FX_FUTURES[ticker]
        n_slices = 13

        # Build strategies
        strategies: Dict[str, List[ExecutionSlice]] = {
            "TWAP": TWAPScheduler(n_slices=n_slices).schedule(num_contracts),
            "VWAP": VWAPScheduler(n_slices=n_slices).schedule(num_contracts),
            "Adaptive": AdaptiveScheduler(kappa=kappa, n_slices=n_slices).schedule(
                num_contracts
            ),
        }

        # Obtain book snapshots
        snapshots: Optional[List[Dict[str, Any]]] = None
        if use_l2:
            snapshots = _fetch_kdb_snapshots(
                ticker, n_slices, kdb_host, kdb_port, date_str, start_time, end_time
            )

        if snapshots is None:
            logger.info(
                "Using synthetic L2 books for %s (%d snapshots).", ticker, n_slices
            )
            snapshots = [
                _generate_synthetic_book(ticker) for _ in range(n_slices)
            ]

        # Run each strategy through the book snapshots
        results: Dict[str, Dict[str, float]] = {}

        for name, schedule in strategies.items():
            total_slippage = 0.0
            total_fill_value = 0.0
            total_qty = 0.0

            for i, slc in enumerate(schedule):
                book = snapshots[i % len(snapshots)]
                qty = max(1, int(round(slc.quantity)))

                fill = self.walk_book(book, qty, side="buy")

                total_slippage += fill["slippage_pips"] * qty
                total_fill_value += fill["avg_fill_price"] * qty
                total_qty += qty

            avg_fill = total_fill_value / total_qty if total_qty > 0 else 0.0
            avg_slippage = total_slippage / total_qty if total_qty > 0 else 0.0

            results[name] = {
                "avg_fill": round(avg_fill, 10),
                "slippage_pips": round(avg_slippage, 4),
                "total_cost": round(total_fill_value, 6),
            }

        return results
