from collections import defaultdict
from functools import partial

import pandas as pd
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from AlphEx.analytics.return_calculation import get_forward_returns_columns, demean_forward_returns, \
    dateindex_geq_alignment, factor_returns


def factor_information_coefficient(factor_data: pd.DataFrame,
                                   group_adjust: bool = False,
                                   by_group: bool = False):
    """
    Computes the cross-sectional Spearman Rank Correlation-based Information Coefficient (IC)
    between factor values and N period forward returns for each period in the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    # def src_ic(group, cols):
    #     factor = group['factor']
    #     ic_series = group[cols].apply(lambda x: stats.spearmanr(x, factor, nan_policy='omit')[0])
    #     return ic_series

    def src_ic(group, cols):
        ranked = group[cols].rank(method='average')
        factor_rank = group['factor'].rank(method='average')
        return ranked.corrwith(factor_rank)

    grouper = ['date']

    if group_adjust:
        factor_data = demean_forward_returns(factor_data, grouper + ['group'])
    if by_group:
        grouper.append('group')

    fwd_cols = get_forward_returns_columns(factor_data.columns)
    ic_func = partial(src_ic, cols=fwd_cols)
    ic = factor_data.groupby(grouper, observed=True, group_keys=False).apply(ic_func)

    return ic


def mean_information_coefficient(factor_data: pd.DataFrame,
                                 group_adjust: bool = False,
                                 by_group: bool = False,
                                 by_time: str | None = None) -> pd.DataFrame:
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole timerange?
    What is the mean IC for for each group, each week?

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, take the mean IC for each group.
    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor_data, group_adjust, by_group)

    grouper = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))
    if by_group:
        grouper.append('group')

    if len(grouper) == 0:
        return ic.mean()
    else:
        return ic.reset_index(level='group').groupby(grouper, observed=True).mean()


def quantile_turnover(quantile_factor: pd.Series, quantile: int, period: int = 1) -> pd.Series:
    """
    Computes the proportion of names in a factor quantile that were not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, asset and factor quantile.
    quantile : int
        Quantile on which to perform turnover analysis.
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quantile_mask  = quantile_factor[quantile_factor == quantile]
    # quantile_sets = quantile_mask.groupby(level=['date']).apply(lambda x: set(x.index.get_level_values('asset')))
    quantile_sets = quantile_mask.index.to_frame(index=False).groupby('date')['asset'].agg(set)

    prev_quantile_sets = quantile_sets.shift(period)

    new_names = (quantile_sets - prev_quantile_sets).dropna()
    total_names = quantile_sets.loc[new_names.index]

    quant_turnover = new_names.map(len) / total_names.map(len)
    quant_turnover.name = quantile
    return quant_turnover


def factor_rank_autocorrelation(factor_data: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Computes the cross-sectional autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor, i.e., how similarly assets are ranked relative to each other compared to a prior date.
    If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of factor values.
    """

    ranks = factor_data.groupby('date')['factor'].rank()

    asset_factor_rank = ranks.reset_index().pivot(index='date', columns='asset', values='factor')

    asset_shifted = asset_factor_rank.shift(period)

    autocorr = asset_factor_rank.corrwith(asset_shifted, axis=1)
    autocorr.name = period
    return autocorr


def factor_alpha_beta(factor_data: pd.DataFrame,
                      returns: pd.DataFrame | None = None,
                      demeaned: bool = True,
                      group_adjust: bool = False,
                      equal_weight: bool = False) -> pd.DataFrame:
    """
    Compute the alpha (excess returns), alpha t-stat (alpha significance),
    and beta (market exposure) of a factor. A regression is run with
    the period wise factor universe mean return as the independent variable
    and mean period wise return from a portfolio weighted by factor values
    as the dependent variable.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    returns : pd.DataFrame, optional
        Period wise factor returns. If this is None then it will be computed
        with 'factor_returns' function and the passed flags: 'demeaned',
        'group_adjust', 'equal_weight'
    demeaned : bool
        Control how to build factor returns used for alpha/beta computation
        -- see AlphEx.analytics.factor_allocation.factor_returns for a full explanation
    group_adjust : bool
        Control how to build factor returns used for alpha/beta computation
        -- see AlphEx.analytics.factor_allocation.factor_returns for a full explanation
    equal_weight : bool, optional
        Control how to build factor returns used for alpha/beta computation
        -- see AlphEx.analytics.factor_allocation.factor_returns for a full explanation

    Returns
    -------
    alpha_beta : pd.DataFrame
        A list containing the alpha, beta, a t-stat(alpha) for the given factor and forward returns.
    """

    if returns is None:
        returns = factor_returns(factor_data, demeaned, group_adjust, equal_weight)

    fwd_ret_columns = get_forward_returns_columns(factor_data.columns)


    universe_return = factor_data.groupby(level='date')[fwd_ret_columns].mean() #.loc[returns.index]
    # returns index might not match!
    aligned_index = dateindex_geq_alignment(factor_dateindex=factor_data.index.unique('date'),
                                            return_dateindex=universe_return.index)
    universe_return = universe_return.loc[aligned_index]
    universe_return.index = returns.index


    if isinstance(returns, pd.Series):
        returns.name = universe_return.columns[0]
        returns = returns.to_frame()

    alpha_beta_stats = defaultdict(dict)
    for period in returns.columns:
        x = universe_return[period].to_numpy()
        y = returns[period].to_numpy()
        x = add_constant(x)

        freq_adjust = pd.Timedelta('252D') / pd.Timedelta(period)
        try:
            reg_fit = OLS(y, x).fit()
            alpha, beta = reg_fit.params
            alpha_ann = (1 + alpha) ** freq_adjust - 1
            alpha_t, beta_t = reg_fit.tvalues
            r2 = reg_fit.rsquared
        except ValueError:
            alpha_ann = beta = alpha_t = beta_t = r2 = np.nan

        alpha_beta_stats['alpha (ann.)'][period] = alpha_ann
        alpha_beta_stats['beta'][period] = beta
        alpha_beta_stats['alpha t-stat'][period] = alpha_t
        # alpha_beta_stats['beta t-stat'][period] = beta_t
        alpha_beta_stats['R²'][period] = r2

    return pd.DataFrame(alpha_beta_stats).T

