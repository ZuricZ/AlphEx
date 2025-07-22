import re
import warnings
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay
from scipy.stats import mode

from AlphEx.analytics.error_handling_utils import NonMatchingTimezoneError
from AlphEx.analytics.factor_allocation import factor_weights
from AlphEx.analytics.time_utils import diff_custom_calendar_timedeltas, timedelta_to_string, infer_trading_calendar
from MetriX.functions import cum_returns


def compute_forward_returns(factor: pd.Series,
                            prices: pd.DataFrame,
                            periods: Iterable[int] = (1, 5, 10),
                            filter_zscore: int | float | None = None,
                            output_cumulative_multiperiod_returns: bool = True) -> pd.DataFrame:
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    output_cumulative_multiperiod_returns : bool, optional
        If True, forward returns columns will contain multi-period cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """

    factor_dateindex = factor.index.unique(level=0)
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")

    freq = infer_trading_calendar(factor_dateindex, prices.index)

    # factor_dateindex = factor_dateindex.intersection(prices.index)
    # if len(factor_dateindex) == 0:
    #     raise ValueError(
    #         "Factor and prices indices don't match: "
    #         "make sure they have the same convention in terms of datetimes and symbol-names"
    #     )

    # filter prices down to unique assets in `factor`
    prices = prices.filter(items=factor.index.unique(level=1))

    forward_return_dict = {}

    for period in sorted(periods):
        if output_cumulative_multiperiod_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period)
        fwd_ret_dateindex = forward_returns.index

        aligned_idx = dateindex_geq_alignment(factor_dateindex, fwd_ret_dateindex)
        forward_returns = forward_returns.loc[aligned_idx]
        forward_returns.index = factor_dateindex

        if filter_zscore is not None:
            # mask = abs(forward_returns - forward_returns.mean()) > (filter_zscore * forward_returns.std())
            # forward_returns[mask] = np.nan
            mean = forward_returns.rolling(window=252, min_periods=90).mean()
            std = forward_returns.rolling(window=252, min_periods=90).std()
            mask = abs(forward_returns - mean) > (filter_zscore * std)
            forward_returns = forward_returns.mask(mask)

        starts = prices.index[:-period]
        ends = prices.index[period:]

        adjusted_deltas = diff_custom_calendar_timedeltas(starts, ends, freq)    # TODO: do we care about freq?
        mode_delta = pd.Series(adjusted_deltas).mode().iloc[0]
        label = timedelta_to_string(mode_delta)

        forward_return_dict[label] = np.concatenate(forward_returns.values)

    df_forward_returns = pd.DataFrame.from_dict(forward_return_dict)
    df_forward_returns.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df_forward_returns = df_forward_returns.reindex(factor.index)

    # df_forward_returns.index.get_level_values(0).unique().freq = freq  # TODO: do we care about freq?
    df_forward_returns.index.set_names(['date', 'asset'], inplace=True)

    return df_forward_returns


def dateindex_geq_alignment(factor_dateindex: pd.DatetimeIndex,
                            return_dateindex: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Aligns a `factor_dateindex` to the nearest corresponding `return_dateindex` where each element in `factor_dateindex`
    is less than or equal to the corresponding element in `return_dateindex`.

    Parameters
    ----------
    factor_dateindex : pd.DatetimeIndex
        The datetime index for the factor data.
    return_dateindex : pd.DatetimeIndex
        The datetime index for the return data.

    Returns
    -------
    pd.DatetimeIndex
        A new DatetimeIndex aligned to `factor_dateindex` based on the condition that each element in `factor_dateindex`
        is less than or equal to the corresponding element in `return_dateindex`.
    """
    # take the first return AT or AFTER the factor print
    idx = np.searchsorted(return_dateindex, factor_dateindex, side='left')
    valid = idx < len(return_dateindex)
    aligned_idx = return_dateindex[idx[valid]]
    return aligned_idx


def backshift_returns_series(series: pd.Series, N: int):
    """
    Shift a multi-indexed series backwards by N observations in the first level.

    This can be used to convert backward-looking returns into a
    forward-returns series.
    """
    idx = series.index
    dates, sids = idx.levels
    date_labels, sid_labels = map(np.array, idx.labels)

    # Output date labels will contain the all but the last N dates.
    new_dates = dates[:-N]

    # Output data will remove the first M rows, where M is the index of the
    # last record with one of the first N dates.
    cutoff = date_labels.searchsorted(N)
    new_date_labels = date_labels[cutoff:] - N
    new_sid_labels = sid_labels[cutoff:]
    new_values = series.values[cutoff:]

    assert new_date_labels[0] == 0

    new_index = pd.MultiIndex(
        levels=[new_dates, sids],
        codes=[new_date_labels, new_sid_labels],
        sortorder=1,
        names=idx.names,
    )

    return pd.Series(data=new_values, index=new_index)


def demean_forward_returns(factor_data: pd.DataFrame, grouper: list | None = None):
    """
    Convert forward returns to returns relative to mean period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a group neutral portfolio constraint and
    thus allows the factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period return for the Technology stocks in our universe
    was 0.5% in the same period, the group adjusted 5 period return for AAPL in this period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = 'date'

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper, observed=True)[cols].transform(lambda x: x - x.mean())

    return factor_data


def get_forward_returns_columns(columns: Iterable[str | int], require_exact_day_multiple: bool = False):
    """
    Utility that detects and returns the columns that are forward returns
    """

    # If exact day multiples are required in the forward return periods,
    # drop all other columns (e.g. drop 3D12h).
    if require_exact_day_multiple:
        pattern = re.compile(r"^(\d+([D]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

        if sum(valid_columns) < len(valid_columns):
            warnings.warn(
                "Skipping return periods that aren't exact multiples of days."
            )
    else:
        pattern = re.compile(r"^(\d+([Dhms]|ms|us|ns]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

    return columns[valid_columns]


def compute_cumulative_returns(returns: pd.Series):
    """
    Computes cumulative returns from simple daily returns.

    Parameters
    ----------
    returns: pd.Series
        pd.Series containing daily factor returns (i.e. '1D' returns).

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-01-05   1.001310
            2015-01-06   1.000805
            2015-01-07   1.001092
            2015-01-08   0.999200
    """

    return cum_returns(returns, starting_value=1)


def rate_of_return(period_ret: pd.DataFrame, base_period: str):
    """
    Convert returns to 'one_period_len' rate of returns: that is the value the
    returns would have every 'one_period_len' if they had grown at a steady
    rate

    Parameters
    ----------
    period_ret: pd.DataFrame
        DataFrame containing returns values with column headings representing
        the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with 'one_period_len' rate of
        returns values.
    """
    period_len = period_ret.name
    conversion_factor = pd.Timedelta(base_period) / pd.Timedelta(period_len)
    return period_ret.add(1).pow(conversion_factor).sub(1)


def std_conversion(period_std: pd.DataFrame, base_period: str):
    """
    one_period_len standard deviation (or standard error) approximation

    Parameters
    ----------
    period_std: pd.DataFrame
        DataFrame containing standard deviation or standard error values
        with column headings representing the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with one-period
        standard deviation/error values.
    """
    period_len = period_std.name
    conversion_factor = pd.Timedelta(period_len) / pd.Timedelta(base_period)
    return period_std / np.sqrt(conversion_factor)


def factor_returns(factor_data: pd.DataFrame,
                   demeaned: bool = True,
                   group_adjust: bool = False,
                   equal_weight: bool = False,
                   by_asset: bool = False) -> pd.DataFrame:
    """
    Computes period wise returns for portfolio weighted by factor
    values.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    by_asset: bool, optional
        If True, returns are reported separately for each asset.

    Returns
    -------
    returns : pd.DataFrame
        Period wise factor returns
    """

    weights = factor_weights(factor_data, demeaned, group_adjust, equal_weight)

    weighted_returns = factor_data[get_forward_returns_columns(factor_data.columns)].multiply(weights, axis=0)

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level='date').sum()

    return returns


def factor_cumulative_returns(factor_data: pd.DataFrame,
                              period: str,
                              long_short: bool = True,
                              group_neutral: bool = False,
                              equal_weight: bool = False,
                              quantiles: Iterable[int] = None,
                              groups: Iterable[str] = None) -> pd.DataFrame:
    """
    Simulate a portfolio using the factor in input and returns the cumulative
    returns of the simulated portfolio

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of portfolio returns
    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details
    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details
    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """
    fwd_ret_cols = get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)

    if quantiles is not None:
        portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(quantiles)]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]

    returns = factor_returns(portfolio_data, long_short, group_neutral, equal_weight)

    return compute_cumulative_returns(returns[period])


def mean_return_by_quantile(factor_data: pd.DataFrame,
                            by_date: bool = False,
                            by_group: bool = False,
                            demeaned: bool = True,
                            group_adjust: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long-short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_returns : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    return_std_errors : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = ['date', 'group']
        factor_data = demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile', 'date']

    if by_group:
        grouper.append('group')

    fwd_return_cols = get_forward_returns_columns(factor_data.columns)
    group_stats = factor_data.groupby(grouper, observed=False)[fwd_return_cols].agg(['mean', 'std', 'count'])

    mean_returns = group_stats.xs('mean', level=1, axis=1)

    if not by_date:
        grouper = ['factor_quantile']
        if by_group:
            grouper.append('group')
        group_stats = mean_returns.groupby(grouper, observed=False).agg(['mean', 'std', 'count'])
        mean_returns = group_stats.xs('mean', level=1, axis=1)

    return_std_errors = group_stats.xs('std', level=1, axis=1) / np.sqrt(group_stats.xs('count', level=1, axis=1))

    return mean_returns, return_std_errors


def mean_return_quantile_spread(mean_returns: pd.DataFrame,
                                upper_quant: int,
                                lower_quant: int,
                                std_error: pd.DataFrame | None = None) -> Tuple[pd.Series, pd.Series]:
    """
    Computes the difference between the mean returns of two quantiles. 
    Optionally, computes the standard error of this difference under ASSUMPTION OF INDEPENDENCE.

    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean period wise returns by quantile. MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : int
        Quantile of mean return from which we wish to subtract lower quantile mean return.
    lower_quant : int
        Quantile of mean return we wish to subtract from upper quantile mean return.
    std_error : pd.DataFrame, optional
        Period wise standard error in mean return by quantile. Takes the same form as mean_returns.

    Returns
    -------
    mean_return_difference : pd.Series
        Period wise difference in quantile returns.
    joint_std_error : pd.Series
        Period wise standard error of the difference in quantile returns under ASSUMPTION OF INDEPENDENCE
        if std_err is None, this will be None
    """

    mean_upper = mean_returns.xs(upper_quant, level='factor_quantile')
    mean_lower = mean_returns.xs(lower_quant, level='factor_quantile')
    mean_return_difference = (mean_upper - mean_lower)

    if std_error is None:
        joint_std_error = None
    else:
        std_upper = std_error.xs(upper_quant, level='factor_quantile')
        std_lower = std_error.xs(lower_quant, level='factor_quantile')
        joint_std_error = np.sqrt(std_upper ** 2 + std_lower ** 2)  # TODO: correlation would require some lifting

    return mean_return_difference, joint_std_error