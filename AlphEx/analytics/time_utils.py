from typing import Iterable, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day, BusinessDay, CustomBusinessDay


# TODO: All of this should be improved


def infer_trading_calendar(factor_idx, prices_idx):
    """
    Infer the trading calendar from factor and price information.

    Parameters
    ----------
    factor_idx : pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices_idx : pd.DatetimeIndex
        The prices datetimes associated with the factor data

    Returns
    -------
    calendar : pd.DateOffset
    """
    full_idx = factor_idx.union(prices_idx).normalize().drop_duplicates().sort_values()

    # Get all unique weekdays that occur in the index
    weekdays = full_idx.dayofweek.unique()
    weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    weekmask = ' '.join([weekday_map[d] for d in sorted(weekdays)])

    # Compute expected dates according to weekday trading
    expected_trading_days = pd.date_range(
        start=full_idx.min(),
        end=full_idx.max(),
        freq=CustomBusinessDay(weekmask=weekmask)
    )

    # Holidays are missing expected days
    holidays = expected_trading_days.difference(full_idx).date.tolist()

    return CustomBusinessDay(weekmask=weekmask, holidays=holidays)


def timedelta_to_string(timedelta: pd.Timedelta) -> str:
    """
    Utility that converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format

    Parameters
    ----------
    timedelta: pd.Timedelta

    Returns
    -------
    string
        string representation of 'timedelta'
    """
    c = timedelta.components
    units = [
        ('D', c.days),
        ('h', c.hours),
        ('m', c.minutes),
        ('s', c.seconds),
        ('ms', c.milliseconds),
        ('us', c.microseconds),
        ('ns', c.nanoseconds)
    ]
    return ''.join(f"{value}{suffix}" for suffix, value in units if value > 0)


def timedelta_strings_to_integers(sequence: Iterable[pd.Timedelta | str]) -> List[int]:
    """
    Converts pandas string representations of timedeltas into integers of days.

    Parameters
    ----------
    sequence : iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(
        input_time: pd.DatetimeIndex | pd.Timestamp, timedelta: pd.Timedelta, freq: pd.DateOffset
) -> pd.DatetimeIndex | pd.Timestamp:
    """
    Add timedelta to 'input' taking into consideration custom frequency, which
    is used to deal with custom calendars, such as a trading calendar

    Parameters
    ----------
    input_time : pd.DatetimeIndex or pd.Timestamp
    timedelta : pd.Timedelta
    freq : pd.DataOffset (CustomBusinessDay, Day or BusinessDay)

    Returns
    -------
    pd.DatetimeIndex or pd.Timestamp
        input + timedelta
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)
    return input_time + freq * days + offset


def diff_custom_calendar_timedeltas(start, end, freq):
    """
    Computes adjusted timedeltas using a custom calendar, preserving intra-day components.

    Parameters
    ----------
    start : array-like of pd.Timestamp
    end : array-like of pd.Timestamp
    freq : CustomBusinessDay (see infer_trading_calendar)
    freq : pd.DataOffset (CustomBusinessDay, Day or BDay)

    Returns
    -------
    pd.Timedelta
        end - start
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")

    weekmask = getattr(freq, 'weekmask', 'Mon Tue Wed Thu Fri')
    holidays = getattr(freq, 'holidays', [])

    # Re-define if traded on weekends
    if isinstance(freq, Day):
        weekmask = 'Mon Tue Wed Thu Fri Sat Sun'

    # Convert to numpy arrays of datetime64[ns]
    starts = np.asarray(start, dtype="datetime64[ns]")
    ends = np.asarray(end, dtype="datetime64[ns]")

    # Raw timedelta between timestamps
    raw_timedeltas = ends - starts

    # Business day difference between dates (as full days)
    business_days = np.busday_count(
        starts.astype("datetime64[D]"),
        ends.astype("datetime64[D]"),
        weekmask=weekmask,
        holidays=holidays
    )

    # Calendar day difference
    calendar_days = (ends.astype("datetime64[D]") - starts.astype("datetime64[D]")).astype(int)

    # Compute difference and subtract non-business days from raw timedelta
    nonbusiness_days = calendar_days - business_days
    adjusted_timedeltas = raw_timedeltas - pd.to_timedelta(nonbusiness_days, unit="D")

    return adjusted_timedeltas