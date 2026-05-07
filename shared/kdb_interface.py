"""
KDB+ IPC interface for FX market-making project.

Provides table creation, bulk writes, and query methods for:
  - fx_rates:  live mid/bid/ask snapshots
  - fx_rfqs:   request-for-quote log
  - fx_fills:  execution / fill log
  - tick_data: CME FX futures MBP-10 order-book snapshots

Falls back gracefully when qpython is not installed, storing data
in in-memory pandas DataFrames so the rest of the project still runs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import qpython; set a flag so every method can branch cleanly.
# ---------------------------------------------------------------------------
try:
    from qpython import qconnection
    from qpython.qtype import QSYMBOL_LIST, QTIMESTAMP_LIST

    HAS_QPYTHON = True
except ImportError:
    HAS_QPYTHON = False
    logger.info("qpython not installed – using in-memory fallback for KDB+ tables")


class KDBInterface:
    """Thin wrapper around a KDB+ process (or in-memory fallback).

    The in-memory fallback (used when no real ``q.exe`` is running) is shared
    at the **class level** so that data written by one ``KDBInterface``
    instance is visible to every subsequent instance in the same Python
    process.  Without this sharing, Step 0c's writes vanish before Step 5b
    reads them, which silently downgrades the L2 book walk to synthetic.
    When a real KDB+ process is connected, ``self.conn is not None`` and the
    fallback is bypassed entirely.
    """

    # Class-level in-memory fallback shared across instances.  This stand-in
    # for the KDB+ tables only matters when ``HAS_QPYTHON`` is False or the
    # connection fails.  When a real KDB+ process is reachable, every read
    # and write goes through ``self.conn`` and this dict is unused.
    _SHARED_FALLBACK: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ init
    def __init__(self, host: str = "localhost", port: int = 5000) -> None:
        """
        Connect to a KDB+ instance.

        Parameters
        ----------
        host : str
            KDB+ server hostname, e.g. ``"localhost"``.
        port : int
            KDB+ server port, e.g. ``5000``.
        """
        self.host = host
        self.port = port
        self.conn = None
        # Bind the shared dict to ``self._fallback`` so existing methods that
        # read ``self._fallback`` continue to work unchanged.  Mutations go
        # through this same dict object, so all instances see them.
        self._fallback = type(self)._SHARED_FALLBACK

        if HAS_QPYTHON:
            try:
                self.conn = qconnection.QConnection(host=host, port=port)
                self.conn.open()
                logger.info("Connected to KDB+ at %s:%s", host, port)
            except Exception as exc:
                logger.warning(
                    "Could not connect to KDB+ at %s:%s (%s) – using in-memory fallback",
                    host,
                    port,
                    exc,
                )
                self.conn = None
        else:
            logger.info("Running without KDB+ (qpython not available)")

    # --------------------------------------------------------- table schemas
    # Q DDL for each table --------------------------------------------------
    _TABLE_SCHEMAS: dict[str, str] = {
        "fx_rates": (
            "fx_rates:([] "
            "timestamp:`timestamp$(); "
            "pair:`symbol$(); "
            "bid:`float$(); "
            "ask:`float$(); "
            "mid:`float$())"
        ),
        "fx_rfqs": (
            "fx_rfqs:([] "
            "timestamp:`timestamp$(); "
            "client:`symbol$(); "
            "pair:`symbol$(); "
            "direction:`symbol$(); "
            "notional:`float$(); "
            "mid:`float$(); "
            "quoted_spread:`float$(); "
            "was_hit:`boolean$())"
        ),
        "fx_fills": (
            "fx_fills:([] "
            "timestamp:`timestamp$(); "
            "pair:`symbol$(); "
            "direction:`symbol$(); "
            "fill_price:`float$(); "
            "notional:`float$(); "
            "strategy:`symbol$())"
        ),
        "tick_data": (
            "tick_data:([] "
            "timestamp:`timestamp$(); "
            "ticker:`symbol$(); "
            + "; ".join(f"bid_px_{i}:`float$()" for i in range(10))
            + "; "
            + "; ".join(f"ask_px_{i}:`float$()" for i in range(10))
            + "; "
            + "; ".join(f"bid_sz_{i}:`long$()" for i in range(10))
            + "; "
            + "; ".join(f"ask_sz_{i}:`long$()" for i in range(10))
            + ")"
        ),
    }

    # Pandas dtype specs used by the in-memory fallback ----------------------
    _FALLBACK_DTYPES: dict[str, dict] = {
        "fx_rates": {
            "timestamp": "datetime64[ns]",
            "pair": "object",
            "bid": "float64",
            "ask": "float64",
            "mid": "float64",
        },
        "fx_rfqs": {
            "timestamp": "datetime64[ns]",
            "client": "object",
            "pair": "object",
            "direction": "object",
            "notional": "float64",
            "mid": "float64",
            "quoted_spread": "float64",
            "was_hit": "bool",
        },
        "fx_fills": {
            "timestamp": "datetime64[ns]",
            "pair": "object",
            "direction": "object",
            "fill_price": "float64",
            "notional": "float64",
            "strategy": "object",
        },
        "tick_data": {
            "timestamp": "datetime64[ns]",
            "ticker": "object",
            **{f"bid_px_{i}": "float64" for i in range(10)},
            **{f"ask_px_{i}": "float64" for i in range(10)},
            **{f"bid_sz_{i}": "int64" for i in range(10)},
            **{f"ask_sz_{i}": "int64" for i in range(10)},
        },
    }

    # ------------------------------------------------------ create_tables
    def create_tables(self) -> None:
        """Create all four FX tables (idempotent)."""
        if self.conn is not None:
            for name, q_expr in self._TABLE_SCHEMAS.items():
                try:
                    self.conn.sendSync(q_expr)
                    logger.info("Created KDB+ table: %s", name)
                except Exception as exc:
                    logger.error("Failed to create table %s: %s", name, exc)
        else:
            for name, dtypes in self._FALLBACK_DTYPES.items():
                if name not in self._fallback:
                    self._fallback[name] = pd.DataFrame(
                        {col: pd.Series(dtype=dt) for col, dt in dtypes.items()}
                    )
                    logger.info("Created in-memory table: %s", name)

    # ------------------------------------------------------ table_counts
    def table_counts(self) -> dict[str, int]:
        """Return ``{table_name: row_count}`` for every managed table.

        Example return value::

            {'fx_rates': 14400, 'fx_rfqs': 312, 'fx_fills': 48, 'tick_data': 864000}
        """
        counts: dict[str, int] = {}
        if self.conn is not None:
            for name in self._TABLE_SCHEMAS:
                try:
                    result = self.conn.sendSync(f"count {name}")
                    counts[name] = int(result)
                except Exception:
                    counts[name] = 0
        else:
            for name in self._FALLBACK_DTYPES:
                counts[name] = len(self._fallback.get(name, []))
        return counts

    # ------------------------------------------------- write_tick_data_bulk
    def write_tick_data_bulk(self, df: pd.DataFrame) -> int:
        """Bulk-insert a DataFrame into the ``tick_data`` table.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``timestamp``, ``ticker``,
            ``bid_px_0..9``, ``ask_px_0..9``, ``bid_sz_0..9``, ``ask_sz_0..9``.

        Returns
        -------
        int
            Number of rows written.  For example, ``86400`` for a full day
            of per-second snapshots.
        """
        required_cols = {"timestamp", "ticker"}
        for prefix in ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_"):
            for i in range(10):
                required_cols.add(f"{prefix}{i}")

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

        n_rows = len(df)

        if self.conn is not None:
            try:
                # Build q insert statement
                self.conn.sendSync(
                    "{[t] `tick_data insert t}",
                    df[sorted(required_cols)].to_dict("list"),
                )
                logger.info("Wrote %d rows to KDB+ tick_data", n_rows)
            except Exception as exc:
                logger.error("KDB+ bulk write failed (%s) – falling back to memory", exc)
                self._ensure_fallback("tick_data")
                self._fallback["tick_data"] = pd.concat(
                    [self._fallback["tick_data"], df[sorted(required_cols)]],
                    ignore_index=True,
                )
        else:
            self._ensure_fallback("tick_data")
            self._fallback["tick_data"] = pd.concat(
                [self._fallback["tick_data"], df[sorted(required_cols)]],
                ignore_index=True,
            )
            logger.info("Wrote %d rows to in-memory tick_data", n_rows)

        return n_rows

    # ------------------------------------------------- tick_data_count
    def tick_data_count(self, ticker: Optional[str] = None) -> int:
        """Return the number of rows in ``tick_data``, optionally filtered.

        Parameters
        ----------
        ticker : str, optional
            If provided, count only rows for this ticker, e.g. ``"6E"``.

        Returns
        -------
        int
            Row count.  For example ``tick_data_count("6E")`` might return
            ``216000`` (one snapshot per second for 6 hours of active trading).
        """
        if self.conn is not None:
            try:
                if ticker is None:
                    return int(self.conn.sendSync("count tick_data"))
                return int(
                    self.conn.sendSync(
                        f'count select from tick_data where ticker=`$"{ticker}"'
                    )
                )
            except Exception:
                return 0
        else:
            tbl = self._fallback.get("tick_data")
            if tbl is None or tbl.empty:
                return 0
            if ticker is None:
                return len(tbl)
            return int((tbl["ticker"] == ticker).sum())

    # -------------------------------------------- get_book_snapshots
    def get_book_snapshots(
        self,
        ticker: str,
        date_str: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
        n_snapshots: int = 100,
    ) -> pd.DataFrame:
        """Return *n_snapshots* evenly-spaced book snapshots for a ticker/day.

        Parameters
        ----------
        ticker : str
            CME ticker symbol, e.g. ``"6E"`` for EUR/USD futures.
        date_str : str
            ISO date, e.g. ``"2025-01-15"``.
        start_time : str
            Start time ``"HH:MM"``, e.g. ``"13:00"`` for London open.
        end_time : str
            End time ``"HH:MM"``, e.g. ``"17:00"`` for NY close.
        n_snapshots : int
            Number of evenly-spaced rows to return, e.g. ``100``.

        Returns
        -------
        pd.DataFrame
            Subset of ``tick_data`` columns with *n_snapshots* rows.
        """
        start_dt = pd.Timestamp(f"{date_str} {start_time}")
        end_dt = pd.Timestamp(f"{date_str} {end_time}")

        if self.conn is not None:
            try:
                q = (
                    f"select from tick_data where ticker=`$\"{ticker}\", "
                    f'timestamp within ({start_dt.isoformat()}; {end_dt.isoformat()})'
                )
                full = pd.DataFrame(self.conn.sendSync(q))
            except Exception as exc:
                logger.warning("KDB+ query failed (%s) – trying fallback", exc)
                full = self._query_fallback(ticker, start_dt, end_dt)
        else:
            full = self._query_fallback(ticker, start_dt, end_dt)

        if full.empty:
            return full

        # Evenly space n_snapshots across the result set
        indices = np.linspace(0, len(full) - 1, min(n_snapshots, len(full)), dtype=int)
        return full.iloc[indices].reset_index(drop=True)

    # ---------------------------------------------------------------- close
    def close(self) -> None:
        """Close the KDB+ connection (no-op for in-memory fallback)."""
        if self.conn is not None:
            try:
                self.conn.close()
                logger.info("KDB+ connection closed")
            except Exception as exc:
                logger.warning("Error closing KDB+ connection: %s", exc)
            finally:
                self.conn = None

    # ------------------------------------------------------------ helpers
    def _ensure_fallback(self, table: str) -> None:
        """Lazily initialise an in-memory table if it doesn't exist yet."""
        if table not in self._fallback:
            dtypes = self._FALLBACK_DTYPES.get(table, {})
            self._fallback[table] = pd.DataFrame(
                {col: pd.Series(dtype=dt) for col, dt in dtypes.items()}
            )

    def _query_fallback(
        self, ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        """Filter the in-memory ``tick_data`` table."""
        tbl = self._fallback.get("tick_data")
        if tbl is None or tbl.empty:
            return pd.DataFrame()
        mask = (
            (tbl["ticker"] == ticker)
            & (tbl["timestamp"] >= start_dt)
            & (tbl["timestamp"] <= end_dt)
        )
        return tbl.loc[mask].copy()

    # -------------------------------------------------------- dunder
    def __repr__(self) -> str:
        backend = "KDB+" if self.conn is not None else "in-memory"
        return f"KDBInterface({self.host}:{self.port}, backend={backend})"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
