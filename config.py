"""
FX Fix-Aware Market Making: W-Shaped Pattern (Krohn, Mueller & Whelan, JoF 2024)
Central configuration module.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FRED API
# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", None)

def get_fred_api_key() -> str | None:
    """Return FRED API key from env var or .env file."""
    key = FRED_API_KEY
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

# ---------------------------------------------------------------------------
# Databento API
# ---------------------------------------------------------------------------
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", None)

def get_databento_api_key() -> str | None:
    """Return Databento API key from env var or .env file."""
    key = DATABENTO_API_KEY
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

# ---------------------------------------------------------------------------
# KDB+ Connection
# ---------------------------------------------------------------------------
KDB_HOST = os.environ.get("KDB_HOST", "localhost")
KDB_PORT = int(os.environ.get("KDB_PORT", "5001"))  # 5001 to avoid collision with rates project

# ---------------------------------------------------------------------------
# FX-Specific Constants
# ---------------------------------------------------------------------------
# CME FX Futures: ticker -> (full name, contract size, tick size, tick value)
FX_FUTURES = {
    "6E": {"name": "EUR/USD", "contract_size": 125_000, "tick_size": 0.00005, "tick_value": 6.25},
    "6B": {"name": "GBP/USD", "contract_size": 62_500, "tick_size": 0.0001, "tick_value": 6.25},
    "6J": {"name": "JPY/USD", "contract_size": 12_500_000, "tick_size": 0.0000005, "tick_value": 6.25},
    "6A": {"name": "AUD/USD", "contract_size": 100_000, "tick_size": 0.0001, "tick_value": 10.00},
}

# FRED series for FX spot rates (USD per foreign currency, except JPY which is JPY per USD)
FRED_FX_SERIES = {
    "EUR/USD": "DEXUSEU",
    "GBP/USD": "DEXUSUK",
    "JPY/USD": "DEXJPUS",
    "AUD/USD": "DEXUSAL",
}

# FRED series for OIS / policy rates
FRED_RATE_SERIES = {
    "USD_SOFR": "SOFR",
    "USD_EFFR": "EFFR",
    "USD_1M": "SOFR30DAYAVG",
    "USD_3M": "SOFR90DAYAVG",
    "EUR_ESTR": "ECBESTRVOLWGTTRMDMNRT",   # ECB €STR
    "GBP_SONIA": "IUDSOIA",                 # SONIA
    "USD_2Y": "DGS2",
    "USD_5Y": "DGS5",
    "USD_10Y": "DGS10",
    "USD_30Y": "DGS30",
}

# Fix times (UTC) -- the three daily FX fixes
FIX_TIMES_UTC = {
    "tokyo":  {"hour": 0, "minute": 55},   # 9:55 JST = 00:55 UTC
    "ecb":    {"hour": 13, "minute": 15},   # 14:15 CET = 13:15 UTC
    "london": {"hour": 16, "minute": 0},    # 16:00 GMT = 16:00 UTC
}

# G9 currency pairs (vs USD)
G9_PAIRS = ["EUR/USD", "GBP/USD", "JPY/USD", "AUD/USD", "CAD/USD", "CHF/USD", "NZD/USD", "NOK/USD", "SEK/USD"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

def setup_logging(name: str = "") -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VALUATION_DATE = "2026-03-05"
DEFAULT_CACHE_TTL_HOURS = 24
BUMP_SIZE_BPS = 1
