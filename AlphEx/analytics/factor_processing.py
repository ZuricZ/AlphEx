from typing import Iterable, Dict, Tuple

import numpy as np
import pandas as pd

from AlphEx.analytics.error_handling_utils import MaxDataLossExceededError, handle_nonunique_bin_edges
from AlphEx.analytics.return_calculation import compute_forward_returns


def format_factor(
        factor: pd.Series,
        forward_returns: pd.DataFrame,
        groupby: pd.Series | Dict | None = None,
        binning_by_group: bool = False,
        quantiles: int = 5,
        bins: int | Iterable[float] | None = None,
        groupby_labels: Dict | None = None,
        max_data_loss_pct: float = 0.35,
        zero_aware: bool = False,
        verbose: bool = True
):
    """
    Formats the factor data, forward return data, and group mappings into a
    DataFrame that contains aligned MultiIndex indices of timestamp and asset.
    The returned data will be formatted to be suitable for AlphaEx functions.

    It is safe to skip a call to this function and still make use of AlphaEx
    functionalities as long as the factor data conforms to the format returned
    from factor_analysis_pipe and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names must follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc.).
        'date' index freq property must be set to a trading calendar
        (pandas DateOffset), see infer_trading_calendar for more details.
        This information is currently used only in cumulative returns
        computation
        ::
            ---------------------------------------
                       |       | 1D  | 5D  | 10D
            ---------------------------------------
                date   | asset |     |     |
            ---------------------------------------
                       | AAPL  | 0.09|-0.01|-0.079
                       ----------------------------
                       | BA    | 0.02| 0.06| 0.020
                       ----------------------------
            2014-01-01 | CMG   | 0.03| 0.09| 0.036
                       ----------------------------
                       | DAL   |-0.02|-0.06|-0.029
                       ----------------------------
                       | LULU  |-0.03| 0.05|-0.009
                       ----------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across groups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analysed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (value-wise) bins to use in factor bucketing.
        Alternately, sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_data_loss_pct : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_data_loss_pct=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
        'quantiles' is None.
    verbose : bool, optional
        If True, the functions prints out progress and data loss percentage.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - forward returns column names follow the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc.)

        - 'date' index freq property (merged_data.index.levels[0].freq) is the
          same as that of the input forward returns data. This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------
    """

    initial_factor_data_length = len(factor.index)

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):  # Convert dict to Series indexed by asset
            asset_index = factor_copy.index.get_level_values('asset')
            missing_assets = set(asset_index) - set(groupby)
            if missing_assets:
                raise KeyError(f"Assets {sorted(missing_assets)} not in group mapping.")

            groupby = pd.Series(groupby).reindex(asset_index, copy=False)
            groupby.index = factor_copy.index  # align to full MultiIndex

        if groupby_labels is not None:
            missing_groups = set(groupby) - set(groupby_labels)
            if missing_groups:
                raise KeyError(f"Groups {sorted(missing_groups)} not in passed group names.")

            groupby = groupby.map(groupby_labels)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()  # TODO: careful

    merged_data_length = len(merged_data.index)

    no_raise = False if max_data_loss_pct == 0 else True
    quantile_data = quantize_factor(
        merged_data,
        quantiles,
        bins,
        binning_by_group,
        no_raise,
        zero_aware
    )

    merged_data['factor_quantile'] = quantile_data

    merged_data = merged_data.dropna()  # TODO: careful

    binning_data_length = len(merged_data.index)

    total_data_loss = (initial_factor_data_length - binning_data_length) / initial_factor_data_length
    forward_data_loss = (initial_factor_data_length - merged_data_length) / initial_factor_data_length
    bin_data_loss = total_data_loss - forward_data_loss

    if verbose:
        print(
            f"Dropped {total_data_loss * 100:.1f}% entries from factor data: "
            f"{forward_data_loss * 100:.1f}% in forward returns computation and "
            f"{bin_data_loss * 100:.1f}% in binning phase "
            "(set max_data_loss_pct=0 to see potentially suppressed Exceptions)."
        )

    if total_data_loss > max_data_loss_pct:
        message = (
            f"max_loss ({max_data_loss_pct * 100:.1f}%) exceeded "
            f"{total_data_loss * 100:.1f}%, consider increasing it."
        )
        raise MaxDataLossExceededError(message)
    elif verbose:
        print(f"max_loss is {max_data_loss_pct * 100:.1f}%, not exceeded: OK!")

    return merged_data


@handle_nonunique_bin_edges
def quantize_factor(
        factor_data: pd.DataFrame,
        quantiles: int | Iterable[float] = 5,
        bins: int | Iterable[float] | None = None,
        by_group: bool = False,
        no_raise: bool = False,
        zero_aware: bool = False
):
    """
    Computes period wise factor quantiles.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (value-wise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, optional
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """
    if not (quantiles is not None and bins is None) or (quantiles is None and bins is not None):
        raise ValueError('Either quantiles or bins should be provided')

    if zero_aware and not isinstance(quantiles, int) or isinstance(bins, int):
        msg = "zero_aware should only be True when quantiles or bins is an integer."
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            # if _quantiles is not None:
            #     if not _zero_aware:
            #         return pd.qcut(x, _quantiles, labels=False) + 1
            #     else:
            #         half_quantiles = _quantiles // 2
            #         pos_quantiles = pd.qcut(x[x >= 0], half_quantiles, labels=False) + half_quantiles + 1
            #         neg_quantiles = pd.qcut(x[x < 0], half_quantiles, labels=False) + 1
            #         return pd.concat([pos_quantiles, neg_quantiles]).reindex(x.index)
            #
            # elif _bins is not None:
            #     if not _zero_aware:
            #         return pd.cut(x, _bins, labels=False) + 1
            #     else:
            #         half_bins = _bins // 2
            #         pos_bins = pd.cut(x[x >= 0], half_bins, labels=False) + half_bins + 1
            #         neg_bins = pd.cut(x[x < 0], half_bins, labels=False) + 1
            #         return pd.concat([pos_bins, neg_bins]).reindex(x.index)
            # else:
            #     return pd.Series(index=x.index)

            if _quantiles is not None:
                method = pd.qcut
                param = _quantiles
            elif _bins is not None:
                method = pd.cut
                param = _bins
            else:
                return pd.Series(index=x.index)  # return NaN

            def apply_method(series, bins, offset):
                return method(series, bins, labels=False) + offset

            if not _zero_aware:
                return apply_method(x, param, offset=1)
            else:
                half_param = param // 2
                pos_result = apply_method(x[x >= 0], half_param, offset=half_param + 1)
                neg_result = apply_method(x[x < 0], half_param, offset=1)
                return pd.concat([pos_result, neg_result]).reindex(x.index)

        except Exception as E:
            if _no_raise:
                return pd.Series(index=x.index)  # return NaN
            raise E

    grouper = ['date'] # factor_data.index.unique(level='date')
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper, observed=True, group_keys=False)['factor'].apply(
        quantile_calc, quantiles, bins, zero_aware, no_raise
    )
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()  # TODO: careful


def factor_analysis_pipe(
        factor: pd.Series,
        prices: pd.DataFrame,
        groupby: pd.Series | None = None,
        binning_by_group: bool = False,
        quantiles: int = 5,
        bins: int | Iterable[float] | None = None,
        periods: Tuple[int] = (1, 5, 10),
        filter_zscore: int | float = 20,
        groupby_labels: Dict | None = None,
        max_data_loss_pct: float = 0.35,
        zero_aware: bool = False,
        output_cumulative_multiperiod: bool = True,
        verbose: bool = True
):
    """
    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for AlphEx functions.

    It is safe to skip a call to this function and still make use of AlphEx
    functionalities as long as the factor data conforms to the format returned
    from factor_analysis_pipe and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of a period your signal was generated so to avoid lookahead
        bias, or delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across groups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analysed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (value-wise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_data_loss_pct : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    output_cumulative_multiperiod : bool, optional
        If True, forward returns columns will contain multi-period cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.
    verbose : bool, optional
        If True, the functions prints out progress and data loss percentage.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    AlphEx.analytics.factor_processing.format_factor
        For use when forward returns are already available.
    """
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        output_cumulative_multiperiod,
    )

    factor_data = format_factor(
        factor, forward_returns,
        groupby=groupby,
        groupby_labels=groupby_labels,
        quantiles=quantiles, bins=bins,
        binning_by_group=binning_by_group,
        max_data_loss_pct=max_data_loss_pct,
        zero_aware=zero_aware,
        verbose=verbose
    )

    return factor_data


