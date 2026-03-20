"""
Date and business-day utilities for FX market-making.

Covers:
  - Day-count conventions (ACT/360, ACT/365, ACT/ACT)
  - FX settlement rules (T+2 standard, T+1 for USD/CAD)
  - US and UK bank holiday calendars (2024-2027)
  - Business-day arithmetic
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Union

DateLike = Union[dt.date, dt.datetime]


# ---------------------------------------------------------------------------
# Day-count conventions
# ---------------------------------------------------------------------------
class DayCountConvention(Enum):
    """Standard day-count conventions used in FX forwards / swaps.

    Examples::

        DayCountConvention.ACT_360   # EUR/USD forward points
        DayCountConvention.ACT_365   # GBP crosses
        DayCountConvention.ACT_ACT   # some exotic pairs
    """

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"


def day_count_fraction(
    start: DateLike,
    end: DateLike,
    convention: DayCountConvention = DayCountConvention.ACT_360,
) -> float:
    """Compute the year fraction between two dates.

    Parameters
    ----------
    start, end : date or datetime
        Period boundaries.
    convention : DayCountConvention
        Day-count rule.

    Returns
    -------
    float
        Year fraction.  For example::

            day_count_fraction(date(2025, 1, 15), date(2025, 7, 15), ACT_360)
            # => 181 / 360 = 0.50278

    Examples
    --------
    >>> from datetime import date
    >>> day_count_fraction(date(2025, 1, 1), date(2025, 7, 1), DayCountConvention.ACT_360)
    0.5027777777777778
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    actual_days = (end_d - start_d).days

    if convention == DayCountConvention.ACT_360:
        return actual_days / 360.0
    elif convention == DayCountConvention.ACT_365:
        return actual_days / 365.0
    elif convention == DayCountConvention.ACT_ACT:
        # Simplified: use actual calendar year length of start year
        year_days = 366 if _is_leap(start_d.year) else 365
        return actual_days / year_days
    else:
        raise ValueError(f"Unknown convention: {convention}")


# ---------------------------------------------------------------------------
# Holiday calendars  (deterministic, 2024-2027)
# ---------------------------------------------------------------------------
# US federal holidays observed by FX markets
_US_HOLIDAYS: set[dt.date] = {
    # 2024
    dt.date(2024, 1, 1),   # New Year's Day
    dt.date(2024, 1, 15),  # MLK Day
    dt.date(2024, 2, 19),  # Presidents' Day
    dt.date(2024, 5, 27),  # Memorial Day
    dt.date(2024, 6, 19),  # Juneteenth
    dt.date(2024, 7, 4),   # Independence Day
    dt.date(2024, 9, 2),   # Labor Day
    dt.date(2024, 10, 14), # Columbus Day
    dt.date(2024, 11, 11), # Veterans Day
    dt.date(2024, 11, 28), # Thanksgiving
    dt.date(2024, 12, 25), # Christmas
    # 2025
    dt.date(2025, 1, 1),
    dt.date(2025, 1, 20),  # MLK Day
    dt.date(2025, 2, 17),  # Presidents' Day
    dt.date(2025, 5, 26),  # Memorial Day
    dt.date(2025, 6, 19),  # Juneteenth
    dt.date(2025, 7, 4),
    dt.date(2025, 9, 1),   # Labor Day
    dt.date(2025, 10, 13), # Columbus Day
    dt.date(2025, 11, 11), # Veterans Day
    dt.date(2025, 11, 27), # Thanksgiving
    dt.date(2025, 12, 25),
    # 2026
    dt.date(2026, 1, 1),
    dt.date(2026, 1, 19),  # MLK Day
    dt.date(2026, 2, 16),  # Presidents' Day
    dt.date(2026, 5, 25),  # Memorial Day
    dt.date(2026, 6, 19),
    dt.date(2026, 7, 3),   # Independence Day observed (July 4 = Sat)
    dt.date(2026, 9, 7),   # Labor Day
    dt.date(2026, 10, 12), # Columbus Day
    dt.date(2026, 11, 11), # Veterans Day
    dt.date(2026, 11, 26), # Thanksgiving
    dt.date(2026, 12, 25),
    # 2027
    dt.date(2027, 1, 1),
    dt.date(2027, 1, 18),  # MLK Day
    dt.date(2027, 2, 15),  # Presidents' Day
    dt.date(2027, 5, 31),  # Memorial Day
    dt.date(2027, 6, 18),  # Juneteenth observed (June 19 = Sat)
    dt.date(2027, 7, 5),   # Independence Day observed (July 4 = Sun)
    dt.date(2027, 9, 6),   # Labor Day
    dt.date(2027, 10, 11), # Columbus Day
    dt.date(2027, 11, 11), # Veterans Day
    dt.date(2027, 11, 25), # Thanksgiving
    dt.date(2027, 12, 24), # Christmas observed (Dec 25 = Sat)
}

# UK bank holidays (England & Wales)
_UK_HOLIDAYS: set[dt.date] = {
    # 2024
    dt.date(2024, 1, 1),
    dt.date(2024, 3, 29),  # Good Friday
    dt.date(2024, 4, 1),   # Easter Monday
    dt.date(2024, 5, 6),   # Early May
    dt.date(2024, 5, 27),  # Spring bank holiday
    dt.date(2024, 8, 26),  # Summer bank holiday
    dt.date(2024, 12, 25),
    dt.date(2024, 12, 26), # Boxing Day
    # 2025
    dt.date(2025, 1, 1),
    dt.date(2025, 4, 18),  # Good Friday
    dt.date(2025, 4, 21),  # Easter Monday
    dt.date(2025, 5, 5),   # Early May
    dt.date(2025, 5, 26),  # Spring bank holiday
    dt.date(2025, 8, 25),  # Summer bank holiday
    dt.date(2025, 12, 25),
    dt.date(2025, 12, 26),
    # 2026
    dt.date(2026, 1, 1),
    dt.date(2026, 4, 3),   # Good Friday
    dt.date(2026, 4, 6),   # Easter Monday
    dt.date(2026, 5, 4),   # Early May
    dt.date(2026, 5, 25),  # Spring bank holiday
    dt.date(2026, 8, 31),  # Summer bank holiday
    dt.date(2026, 12, 25),
    dt.date(2026, 12, 28), # Boxing Day observed
    # 2027
    dt.date(2027, 1, 1),
    dt.date(2027, 3, 26),  # Good Friday
    dt.date(2027, 3, 29),  # Easter Monday
    dt.date(2027, 5, 3),   # Early May
    dt.date(2027, 5, 31),  # Spring bank holiday
    dt.date(2027, 8, 30),  # Summer bank holiday
    dt.date(2027, 12, 27), # Christmas observed
    dt.date(2027, 12, 28), # Boxing Day observed
}

# Combined set used for FX settlement (both centres must be open)
_FX_HOLIDAYS: set[dt.date] = _US_HOLIDAYS | _UK_HOLIDAYS


# ---------------------------------------------------------------------------
# Business-day helpers
# ---------------------------------------------------------------------------
def is_business_day(date: DateLike) -> bool:
    """Check whether *date* is a valid FX business day.

    A business day is a weekday that is not a US or UK bank holiday.

    Parameters
    ----------
    date : date or datetime
        The date to check, e.g. ``date(2025, 12, 25)`` (Christmas -> False).

    Returns
    -------
    bool
        ``True`` if FX markets are open.

    Examples
    --------
    >>> is_business_day(dt.date(2025, 1, 20))  # MLK Day
    False
    >>> is_business_day(dt.date(2025, 1, 21))  # Tuesday
    True
    """
    d = _to_date(date)
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in _FX_HOLIDAYS


def next_business_day(date: DateLike) -> dt.date:
    """Return the next FX business day on or after *date*.

    Parameters
    ----------
    date : date or datetime
        Starting date.  For example ``date(2025, 12, 25)`` (Thursday,
        Christmas) returns ``date(2025, 12, 29)`` (Monday).

    Returns
    -------
    dt.date
    """
    d = _to_date(date)
    while not is_business_day(d):
        d += dt.timedelta(days=1)
    return d


def _advance_business_days(date: DateLike, n: int) -> dt.date:
    """Advance *n* business days from *date* (date itself is NOT counted)."""
    d = _to_date(date)
    count = 0
    while count < n:
        d += dt.timedelta(days=1)
        if is_business_day(d):
            count += 1
    return d


def business_days_between(start: DateLike, end: DateLike) -> int:
    """Count FX business days in the half-open interval ``[start, end)``.

    Parameters
    ----------
    start, end : date or datetime

    Returns
    -------
    int
        Number of business days.  For example::

            business_days_between(date(2025, 1, 6), date(2025, 1, 10))
            # Mon 6 Jan -> Fri 10 Jan  =>  4 business days

    Examples
    --------
    >>> business_days_between(dt.date(2025, 1, 6), dt.date(2025, 1, 10))
    4
    """
    s = _to_date(start)
    e = _to_date(end)
    if s >= e:
        return 0
    count = 0
    current = s
    while current < e:
        if is_business_day(current):
            count += 1
        current += dt.timedelta(days=1)
    return count


# ---------------------------------------------------------------------------
# FX value-date (settlement)
# ---------------------------------------------------------------------------
# Pairs that settle T+1 instead of the standard T+2
_T1_PAIRS: set[str] = {"USDCAD", "CADUSD"}


def fx_value_date(trade_date: DateLike, pair: str = "EURUSD") -> dt.date:
    """Compute the FX spot value (settlement) date.

    Standard FX settles T+2; USD/CAD settles T+1.

    Parameters
    ----------
    trade_date : date or datetime
        The trade date, e.g. ``date(2025, 1, 15)`` (Wednesday).
    pair : str
        Currency pair, e.g. ``"EURUSD"``.  Use ``"USDCAD"`` for T+1.

    Returns
    -------
    dt.date
        Settlement date.  For example::

            fx_value_date(date(2025, 1, 15), "EURUSD")
            # Wed 15 Jan + 2 bd = Fri 17 Jan 2025

            fx_value_date(date(2025, 1, 15), "USDCAD")
            # Wed 15 Jan + 1 bd = Thu 16 Jan 2025

    Examples
    --------
    >>> fx_value_date(dt.date(2025, 1, 15), "EURUSD")
    datetime.date(2025, 1, 17)
    >>> fx_value_date(dt.date(2025, 1, 15), "USDCAD")
    datetime.date(2025, 1, 16)
    """
    normalised = pair.upper().replace("/", "").replace("_", "")
    tenor = 1 if normalised in _T1_PAIRS else 2
    return _advance_business_days(trade_date, tenor)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_date(d: DateLike) -> dt.date:
    """Coerce datetime -> date."""
    if isinstance(d, dt.datetime):
        return d.date()
    return d


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
