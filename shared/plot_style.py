"""
JPM-style institutional plotting utilities for FX market-making.

Sets matplotlib rcParams for clean, presentation-ready charts with a
colour palette matching JPMorgan research publications.

Usage::

    from shared.plot_style import set_jpm_style, JPM_COLORS, format_pips
    set_jpm_style()

    fig, ax = plt.subplots()
    ax.plot(x, y, color=JPM_COLORS["blue"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format_pips(v)))
"""

from __future__ import annotations

from typing import Union

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
JPM_COLORS: dict[str, str] = {
    "blue": "#003366",
    "red": "#CC0000",
    "green": "#006633",
    "gold": "#CC9900",
    "gray": "#666666",
}

# Ordered list for cycling through series
JPM_CYCLE = [
    JPM_COLORS["blue"],
    JPM_COLORS["red"],
    JPM_COLORS["green"],
    JPM_COLORS["gold"],
    JPM_COLORS["gray"],
]


# ---------------------------------------------------------------------------
# Style setter
# ---------------------------------------------------------------------------
def set_jpm_style() -> None:
    """Apply JPM-style rcParams to matplotlib.

    Call once at the top of a notebook or script.  Sets:
      - White background, minimal spines
      - Helvetica-like font (``Arial`` as cross-platform fallback)
      - ``JPM_CYCLE`` as the default colour cycle
      - Grid styling, tick formatting, legend positioning

    Example::

        set_jpm_style()
        # All subsequent plt.plot() / fig, ax = plt.subplots() calls
        # will use the institutional style automatically.
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return  # graceful no-op if matplotlib not installed

    params = {
        # Figure
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": JPM_COLORS["gray"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.prop_cycle": mpl.cycler(color=JPM_CYCLE),
        # Grid
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.loc": "best",
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        # Lines
        "lines.linewidth": 1.5,
        "lines.antialiased": True,
        # Savefig
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
    mpl.rcParams.update(params)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_bps(value: float, decimals: int = 1) -> str:
    """Format a decimal value as basis points.

    Parameters
    ----------
    value : float
        Raw decimal, e.g. ``0.00245`` (= 2.45 bps).
    decimals : int
        Decimal places in output.

    Returns
    -------
    str
        For example ``format_bps(0.00245)`` returns ``"2.5 bps"``.

    Examples
    --------
    >>> format_bps(0.00245)
    '2.5 bps'
    >>> format_bps(0.015, decimals=0)
    '150 bps'
    """
    return f"{value * 10_000:.{decimals}f} bps"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal as a percentage.

    Parameters
    ----------
    value : float
        Raw decimal, e.g. ``0.0532`` (= 5.32 %).

    Returns
    -------
    str
        For example ``format_pct(0.0532)`` returns ``"5.32%"``.

    Examples
    --------
    >>> format_pct(0.0532)
    '5.32%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_ccy(value: float, currency: str = "USD", decimals: int = 0) -> str:
    """Format a monetary amount with currency prefix.

    Parameters
    ----------
    value : float
        Amount, e.g. ``1_250_000.0``.
    currency : str
        ISO currency code.
    decimals : int
        Decimal places.

    Returns
    -------
    str
        For example ``format_ccy(1_250_000)`` returns ``"USD 1,250,000"``.

    Examples
    --------
    >>> format_ccy(1_250_000)
    'USD 1,250,000'
    >>> format_ccy(42_500.75, "EUR", 2)
    'EUR 42,500.75'
    """
    return f"{currency} {value:,.{decimals}f}"


def format_pips(value: float, decimals: int = 1) -> str:
    """Format a price difference in FX pips (1 pip = 0.0001 for most pairs).

    One pip equals the fourth decimal place for major pairs (EUR/USD,
    GBP/USD, AUD/USD) and the second decimal place for JPY pairs.  This
    function uses the standard 0.0001 convention; for JPY pairs, pass
    the value already in the right units or adjust outside.

    Parameters
    ----------
    value : float
        Price difference in decimal, e.g. ``0.00032`` (= 3.2 pips).
    decimals : int
        Decimal places in output.

    Returns
    -------
    str
        For example ``format_pips(0.00032)`` returns ``"3.2 pips"``.

    Examples
    --------
    >>> format_pips(0.00032)
    '3.2 pips'
    >>> format_pips(0.0015, decimals=0)
    '15 pips'
    >>> format_pips(-0.00008, decimals=2)
    '-0.80 pips'
    """
    pips = value * 10_000  # 1 pip = 0.0001
    return f"{pips:.{decimals}f} pips"
