"""
Validation helpers.
Given a predicted date string and the original conditions, checks whether
all four conditions are satisfied. This is used as the evaluation metric
(condition satisfaction rate) since multiple dates can be correct.
"""

from __future__ import annotations
import datetime
from typing import Tuple


MONTH_MAP = {
    "JAN": 1,  "FEB": 2,  "MAR": 3,  "APR": 4,
    "MAY": 5,  "JUN": 6,  "JUL": 7,  "AUG": 8,
    "SEP": 9,  "OCT": 10, "NOV": 11, "DEC": 12,
}

DAY_MAP = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6,
}


def is_leap_year(year: int) -> bool:
    """
    Standard Gregorian leap-year rule:
      - divisible by 4
      - EXCEPT century years, which must also be divisible by 400
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def parse_date(date_str: str) -> Tuple[int, int, int] | None:
    """Parse 'dd-mm-yyyy' and return (dd, mm, yyyy) ints, or None if invalid."""
    try:
        parts = date_str.strip().split("-")
        dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
        datetime.date(yyyy, mm, dd)   # validates calendar correctness
        return dd, mm, yyyy
    except Exception:
        return None


def check_conditions(
    date_str: str,
    day_cond: str,
    month_cond: str,
    leap_cond: str,
    decade_cond: str,
) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Returns (valid, day_ok, month_ok, leap_ok, decade_ok).
    'valid' is False if date_str cannot be parsed as a real calendar date.
    """
    parsed = parse_date(date_str)
    if parsed is None:
        return False, False, False, False, False

    dd, mm, yyyy = parsed
    date_obj = datetime.date(yyyy, mm, dd)

    # Day-of-week: Monday=0, …, Sunday=6
    day_ok    = date_obj.weekday() == DAY_MAP.get(day_cond, -1)
    month_ok  = mm == MONTH_MAP.get(month_cond, -1)
    leap_ok   = is_leap_year(yyyy) == (leap_cond == "True")
    decade_ok = (yyyy // 10) == int(decade_cond)

    return True, day_ok, month_ok, leap_ok, decade_ok


def all_conditions_met(
    date_str: str,
    day_cond: str,
    month_cond: str,
    leap_cond: str,
    decade_cond: str,
) -> bool:
    """Convenience wrapper – True only when all four conditions pass."""
    valid, day_ok, month_ok, leap_ok, decade_ok = check_conditions(
        date_str, day_cond, month_cond, leap_cond, decade_cond
    )
    return valid and day_ok and month_ok and leap_ok and decade_ok
