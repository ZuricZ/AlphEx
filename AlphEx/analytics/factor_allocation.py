import pandas as pd


def factor_weights(
        factor_data: pd.DataFrame,
        demeaned: bool = True,
        group_adjust: bool = False,
        equal_weight: bool = False
):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted
        If demeaned is True then the factor universe will be split in two
        equal sized groups, top assets with positive weights and bottom assets
        with negative weights

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group, _demeaned, _equal_weight):

        if _equal_weight:
            weights = pd.Series(0.0, index=group.index)

            # top assets positive weights, bottom ones negative if demeaned
            centered = group - group.median() if _demeaned else group

            positive_mask = centered > 0
            group[positive_mask] = 1.0
            negative_mask = centered < 0
            group[negative_mask] = -1.0

            if _demeaned:
                # positive weights must equal negative weights
                if positive_mask.any():
                    weights[positive_mask] /= positive_mask.sum()
                if negative_mask.any():
                    weights[negative_mask] /= negative_mask.sum()

            return weights / weights.abs().sum()  # normalise to 1


        if _demeaned:
            group = group - group.mean()

        return group / group.abs().sum()

    grouper = ['date']  # factor_data.index.get_level_values('date')
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper, observed=True, group_keys=False)['factor'].apply(to_weights, demeaned, equal_weight)

    if group_adjust:
        weights = weights.groupby(level='date', observed=True, group_keys=False).apply(to_weights, False, False)

    return weights


# def positions(weights: pd.Series, period: str | pd.Timedelta, freq: pd.DateOffset | None = None) -> pd.DataFrame:
#     """
#     Builds net position values time series, the portfolio percentage invested in each position.
#
#     Parameters
#     ----------
#     weights: pd.Series
#         pd.Series containing factor weights, the index contains timestamps at
#         which the trades are computed and the values correspond to assets
#         weights
#         - see factor_weights for more details
#     period: pandas.Timedelta or string
#         Assets holding period (1 day, 2 mins, 3 hours etc). It can be a
#         Timedelta or a string in the format accepted by Timedelta constructor
#         ('1 days', '1D', '30m', '3h', '1D1h', etc)
#     freq : pandas DateOffset, optional
#         Used to specify a particular trading calendar. If not present
#         weights.index.freq will be used
#
#     Returns
#     -------
#     pd.DataFrame
#         Assets positions series, datetime on index, assets on columns.
#         Example:
#             index                 'AAPL'         'MSFT'          cash
#             2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
#             2004-01-09 15:30:00       0.00       -16012.9930     411.5585
#             2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
#             2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
#             2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
#     """
#
#     weights = weights.unstack()
#
#     if not isinstance(period, pd.Timedelta):
#         period = pd.Timedelta(period)
#
#     if freq is None:
#         freq = weights.index.freq
#
#     if freq is None:
#         freq = BDay()
#         warnings.warn("'freq' not set, using business day calendar", UserWarning)
#
#
#     # weights index contains factor computation timestamps, then add returns
#     # timestamps too (factor timestamps + period) and save them to 'full_idx'
#     # 'full_idx' index will contain an entry for each point in time the weights
#     # change and hence they have to be re-computed
#
#     trades_idx = weights.index.copy()
#     returns_idx = add_custom_calendar_timedelta(trades_idx, period, freq)  # TODO
#     weights_idx = trades_idx.union(returns_idx)  # TODO
#
#
#     # Compute portfolio weights for each point in time contained in the index
#     portfolio_weights = pd.DataFrame(index=weights_idx, columns=weights.columns)
#     active_weights = []
#
#     for curr_time in weights_idx:
#         # fetch new weights that become available at curr_time and store them in active weights
#         if curr_time in weights.index:
#             assets_weights = weights.loc[curr_time]
#             expire_ts = add_custom_calendar_timedelta(curr_time, period, freq)
#             active_weights.append((expire_ts, assets_weights))
#
#         # remove expired entry in active_weights (older than 'period')
#         if active_weights:
#             expire_ts, assets_weights = active_weights[0]
#             if expire_ts <= curr_time:
#                 active_weights.pop(0)
#
#         if not active_weights:
#             continue
#
#         # Compute total weights for curr_time and store them
#         tot_weights = [w for (ts, w) in active_weights]
#         tot_weights = pd.concat(tot_weights, axis=1)
#         tot_weights = tot_weights.sum(axis=1)
#         tot_weights /= tot_weights.abs().sum()
#
#         portfolio_weights.loc[curr_time] = tot_weights
#
#     return portfolio_weights.fillna(0)
#
#
# def factor_positions(
#         factor_data: pd.DataFrame,
#         period: str,
#         long_short: bool = True,
#         group_neutral: bool = False,
#         equal_weight: bool = False,
#         quantiles: bool = None,
#         groups: bool = None
# ) -> pd.DataFrame:
#     """
#     Simulate a portfolio using the factor in input and returns the assets
#     positions as percentage of the total portfolio.
#
#     Parameters
#     ----------
#     factor_data : pd.DataFrame - MultiIndex
#         A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
#         containing the values for a single alpha factor, forward returns for
#         each period, the factor quantile/bin that factor value belongs to,
#         and (optionally) the group the asset belongs to.
#         - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
#     period : string
#         'factor_data' column name corresponding to the 'period' returns to be
#         used in the computation of portfolio returns
#     long_short : bool, optional
#         if True then simulates a dollar neutral long-short portfolio
#         - see performance.create_pyfolio_input for more details
#     group_neutral : bool, optional
#         If True then simulates a group neutral portfolio
#         - see performance.create_pyfolio_input for more details
#     equal_weight : bool, optional
#         Control the assets weights:
#         - see performance.create_pyfolio_input for more details.
#     quantiles: sequence[int], optional
#         Use only specific quantiles in the computation. By default all
#         quantiles are used
#     groups: sequence[string], optional
#         Use only specific groups in the computation. By default all groups
#         are used
#
#     Returns
#     -------
#     assets positions : pd.DataFrame
#         Assets positions series, datetime on index, assets on columns.
#         Example:
#             index                 'AAPL'         'MSFT'          cash
#             2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
#             2004-01-09 15:30:00       0.00       -16012.9930     411.5585
#             2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
#             2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
#             2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
#     """
#     fwd_ret_cols = get_forward_returns_columns(factor_data.columns)
#
#     if period not in fwd_ret_cols:
#         raise ValueError("Period '%s' not found" % period)
#
#     todrop = list(fwd_ret_cols)
#     todrop.remove(period)
#     portfolio_data = factor_data.drop(todrop, axis=1)
#
#     if quantiles is not None:
#         portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(quantiles)]
#
#     if groups is not None:
#         portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]
#
#     weights = factor_weights(portfolio_data, long_short, group_neutral, equal_weight)
#
#     return positions(weights, period)
#
#
# def create_pyfolio_input(factor_data,
#                          period,
#                          capital=None,
#                          long_short=True,
#                          group_neutral=False,
#                          equal_weight=False,
#                          quantiles=None,
#                          groups=None,
#                          benchmark_period='1D'):
#     """
#     Simulate a portfolio using the input factor and returns the portfolio
#     performance data properly formatted for Pyfolio analysis.
#
#     For more details on how this portfolio is built see:
#     - performance.cumulative_returns (how the portfolio returns are computed)
#     - performance.factor_weights (how assets weights are computed)
#
#     Parameters
#     ----------
#     factor_data : pd.DataFrame - MultiIndex
#         A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
#         containing the values for a single alpha factor, forward returns for
#         each period, the factor quantile/bin that factor value belongs to,
#         and (optionally) the group the asset belongs to.
#         - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
#     period : string
#         'factor_data' column name corresponding to the 'period' returns to be
#         used in the computation of portfolio returns
#     capital : float, optional
#         If set, then compute 'positions' in dollar amount instead of percentage
#     long_short : bool, optional
#         if True enforce a dollar neutral long-short portfolio: asset weights
#         will be computed by demeaning factor values and dividing by the sum of
#         their absolute value (achieving gross leverage of 1) which will cause
#         the portfolio to hold both long and short positions and the total
#         weights of both long and short positions will be equal.
#         If False the portfolio weights will be computed dividing the factor
#         values and by the sum of their absolute value (achieving gross
#         leverage of 1). Positive factor values will generate long positions and
#         negative factor values will produce short positions so that a factor
#         with only positive values will result in a long only portfolio.
#     group_neutral : bool, optional
#         If True simulates a group neutral portfolio: the portfolio weights
#         will be computed so that each group will weigh the same.
#         if 'long_short' is enabled the factor values demeaning will occur on
#         the group level resulting in a dollar neutral, group neutral,
#         long-short portfolio.
#         If False group information will not be used in weights computation.
#     equal_weight : bool, optional
#         if True the assets will be equal-weighted. If long_short is True then
#         the factor universe will be split in two equal sized groups with the
#         top assets in long positions and bottom assets in short positions.
#         if False the assets will be factor-weighed, see 'long_short' argument
#     quantiles: sequence[int], optional
#         Use only specific quantiles in the computation. By default all
#         quantiles are used
#     groups: sequence[string], optional
#         Use only specific groups in the computation. By default all groups
#         are used
#     benchmark_period : string, optional
#         By default benchmark returns are computed as the factor universe mean
#         daily returns but 'benchmark_period' allows to choose a 'factor_data'
#         column corresponding to the returns to be used in the computation of
#         benchmark returns. More generally benchmark returns are computed as the
#         factor universe returns traded at 'benchmark_period' frequency, equal
#         weighting and long only
#
#
#     Returns
#     -------
#      returns : pd.Series
#         Daily returns of the strategy, noncumulative.
#          - Time series with decimal returns.
#          - Example:
#             2015-07-16    -0.012143
#             2015-07-17    0.045350
#             2015-07-20    0.030957
#             2015-07-21    0.004902
#
#      positions : pd.DataFrame
#         Time series of dollar amount (or percentage when 'capital' is not
#         provided) invested in each position and cash.
#          - Days where stocks are not held can be represented by 0.
#          - Non-working capital is labelled 'cash'
#          - Example:
#             index         'AAPL'         'MSFT'          cash
#             2004-01-09    13939.3800     -14012.9930     711.5585
#             2004-01-12    14492.6300     -14624.8700     27.1821
#             2004-01-13    -13853.2800    13653.6400      -43.6375
#
#
#      benchmark : pd.Series
#         Benchmark returns computed as the factor universe mean daily returns.
#
#     """
#
#
#     # Build returns:
#     # we don't know the frequency at which the factor returns are computed but
#     # pyfolio wants daily returns. So we compute the cumulative returns of the
#     # factor, then resample it at 1 day frequency and finally compute daily
#     # returns
#
#     cumrets = factor_cumulative_returns(factor_data,
#                                         period,
#                                         long_short,
#                                         group_neutral,
#                                         equal_weight,
#                                         quantiles,
#                                         groups)
#     cumrets = cumrets.resample('1D').last().fillna(method='ffill')
#     returns = cumrets.pct_change().fillna(0)
#
#
#     # Build positions. As pyfolio asks for daily position we have to resample
#     # the positions returned by 'factor_positions' at 1 day frequency and
#     # recompute the weights so that the sum of daily weights is 1.0
#
#     positions = factor_positions(factor_data,
#                                  period,
#                                  long_short,
#                                  group_neutral,
#                                  equal_weight,
#                                  quantiles,
#                                  groups)
#     positions = positions.resample('1D').sum().fillna(method='ffill')
#     positions = positions.div(positions.abs().sum(axis=1), axis=0).fillna(0)
#     positions['cash'] = 1. - positions.sum(axis=1)
#
#     # transform percentage positions to dollar positions
#     if capital is not None:
#         positions = positions.mul(
#             cumrets.reindex(positions.index) * capital, axis=0)
#
#     # Build benchmark returns as the factor universe mean returns traded at
#     # 'benchmark_period' frequency
#
#     fwd_ret_cols = get_forward_returns_columns(factor_data.columns)
#     if benchmark_period in fwd_ret_cols:
#         benchmark_data = factor_data.copy()
#         # make sure no negative positions
#         benchmark_data['factor'] = benchmark_data['factor'].abs()
#         benchmark_rets = factor_cumulative_returns(benchmark_data,
#                                                    benchmark_period,
#                                                    long_short=False,
#                                                    group_neutral=False,
#                                                    equal_weight=True)
#         benchmark_rets = benchmark_rets.resample('1D').last().fillna(method='ffill')
#         benchmark_rets = benchmark_rets.pct_change().fillna(0)
#         benchmark_rets.name = 'benchmark'
#     else:
#         benchmark_rets = None
#
#     return returns, positions, benchmark_rets
