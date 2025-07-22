import warnings
from typing import Any, Literal
from functools import partial

import numpy as np
import pandas as pd

from AlphEx.analytics.return_calculation import compute_cumulative_returns, dateindex_geq_alignment


def slice_returns_by_event(factor: pd.DataFrame,
                           returns: pd.DataFrame,
                           before: int,
                           after: int,
                           input_cumulative_multiperiod_returns: bool = False,
                           mean_by_date: bool = False,
                           demean_by: pd.DataFrame | None = None,
                           align_return_index: bool = True) -> pd.DataFrame:
    """
    A date and equity pair is extracted from each index row in the factor dataframe.
    For each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame

    Parameters
    ----------
    factor: pd.DataFrame
        DataFrame with at least date and equity as index, the columns are irrelevant
    returns: pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets in the
        columns. Returns data should span the factor analysis time period
        plus/minus an additional buffer window corresponding to after/before
        period parameters. Should be indexed by the frequency of the before/after period!
    before: int
        How many returns to load before factor date
    after: int
        How many returns to load after factor date
    input_cumulative_multiperiod_returns: bool, optional
        Whether the input returns are multi-period cumulative (for multi-period returns). If False the given
        returns are assumed to be over a single period.
    mean_by_date: bool, optional
        If True, compute mean returns for each date and return that
        instead of a return series for each asset
    demean_by: pd.DataFrame, optional
        DataFrame with at least date and equity as index, the columns are
        irrelevant. For each date a list of equities is extracted from
        'demean_by' index and used as universe to compute demeaned mean
        returns (long-short portfolio)
    align_return_index: bool, optional
        Used in case return dateindex is different from the factor dateindex.
        When True, this aligns in a way so that the return event is AT or AFTER factor event.
        This results in data loss for before/after period in the beginning/end of the return dataset.
    Returns
    -------
    aligned_returns : pd.DataFrame
        Dataframe containing returns series for each factor aligned to the same index: -before to after
    """
    if not input_cumulative_multiperiod_returns:
        returns = returns.apply(compute_cumulative_returns, axis=0)

    # Align in case return data has inherent latency compared to factor data
    if align_return_index:
        factor_dateindex = factor.index.unique(level='date')
        aligned_index = dateindex_geq_alignment(factor_dateindex=factor_dateindex, return_dateindex=returns.index)
        returns = returns.loc[aligned_index]
        returns.index = factor_dateindex

    frames = []
    for i in range(-before, after + 1):
        shifted = returns.shift(-i).copy()
        shifted.columns = pd.MultiIndex.from_product([[i], returns.columns])
        frames.append(shifted)

    returns_sliced_by_events = pd.concat(frames, axis=1).T.unstack(level=1)

    # Subtract the mean of assets present in demean_by index
    if demean_by is not None:
        group_mean_returns = returns_sliced_by_events.reindex(demean_by.index, axis=1).T.groupby('date').mean().T
        demean_dates = returns_sliced_by_events.columns.get_level_values(0)
        group_mean_returns = group_mean_returns.reindex(demean_dates, axis=1)
        returns_sliced_by_events = returns_sliced_by_events.sub(group_mean_returns.to_numpy(), axis=0)

    # Slice the assets present in the factor
    returns_sliced_by_events = returns_sliced_by_events.reindex(factor.index, axis=1)

    if mean_by_date:
      returns_sliced_by_events = returns_sliced_by_events.T.groupby(level=0).mean().T

    return returns_sliced_by_events


def average_cumulative_return_by_quantile(factor_data: pd.DataFrame,
                                          returns: pd.DataFrame,
                                          periods_before: int = 10,
                                          periods_after: int = 15,
                                          demeaned: bool = True,
                                          group_adjust: bool = False,
                                          by_group: bool = False,
                                          align_return_index: bool = True,
                                          ) -> pd.DataFrame:
    """
    Plots average cumulative returns by factor quantiles in the period range
    defined by -periods_before to periods_after

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in AlphEx.analytics.factor_processing.factor_analysis_pipe
    returns : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets in the
        columns. Returns data should span the factor analysis time period
        plus/minus an additional buffer window corresponding to periods_after/
        periods_before parameters.
    periods_before : int, optional
        How many periods before factor to plot
    periods_after  : int, optional
        How many periods after factor to plot
    demeaned : bool, optional
        Compute demeaned mean returns (long-short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level (group neutral portfolio)
    by_group : bool
        If True, compute cumulative returns separately for each group
    align_return_index: bool, optional
        Used in case return dateindex is different from the factor dateindex.
        When True, this aligns in a way so that the return event is AT or AFTER factor event.
        This results in data loss for before/after period in the beginning/end of the return dataset.

    Returns
    -------
    cumulative returns and std deviation : pd.DataFrame
        A MultiIndex DataFrame indexed by quantile (level 0) and mean/std
        (level 1) and the values on the columns in range from
        -periods_before to periods_after
        If by_group=True the index will have an additional 'group' level:

            ---------------------------------------------------
                        |       | -2  | -1  |  0  |  1  | ...
            ---------------------------------------------------
              quantile  |       |     |     |     |     |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 1      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 2      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                ...     |                 ...
            ---------------------------------------------------
    """

    event_aligned_returns = partial(
        slice_returns_by_event,
        returns=returns,
        before=periods_before,
        after=periods_after,
        input_cumulative_multiperiod_returns=True,  # Assumes cumulative returns for multiperiod returns
        mean_by_date=True,
        align_return_index=align_return_index
    )

    # Select appropriate analysis
    study = EventReturnStudy.create(by_group, group_adjust, demeaned)

    if by_group:
        return study.analyze(
            cumulative_return_func=event_aligned_returns,
            factor_data=factor_data,
            group_adjust=group_adjust,
            demeaned=demeaned
        )
    elif group_adjust:
        return study.analyze(
            cumulative_return_func=event_aligned_returns,
            factor_data=factor_data
        )
    elif demeaned:
        return study.analyze(
            cumulative_return_func=event_aligned_returns,
            factor_data=factor_data
        )
    else:
        return study.analyze(
            cumulative_return_func=event_aligned_returns,
            factor_data=factor_data,
            demean_by=None
        )


class EventReturnStudy:
    """
    Factory for cumulative return analysis strategies
    """
    @staticmethod
    def create(by_group: bool, group_adjust: bool, demeaned: bool):
        if by_group:
            return GroupedStudy()
        elif group_adjust:
            return GroupAdjustedStudy()
        elif demeaned:
            return DemeanedStudy()
        return BaseEventReturnStudy()


class BaseEventReturnStudy:
    """
    Compute quantile cumulative returns for the full factor_data
    Align cumulative return from different dates to the same index then compute mean and std
    """
    def analyze(self, cumulative_return_func, factor_data, demean_by=None, group_adjust=None, demeaned=None):
        factor_quantile = factor_data['factor_quantile']
        return factor_quantile.groupby(factor_quantile).apply(
            self._compute_quantile_stats,
            cumulative_return_func=cumulative_return_func,
            demean_by=demean_by
        )

    @staticmethod
    def _compute_quantile_stats(quantile_series, cumulative_return_func, demean_by):
        quantile_returns = cumulative_return_func(factor=quantile_series, demean_by=demean_by)
        quantile_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        return pd.DataFrame({
            'mean': quantile_returns.mean(skipna=True, axis=1),
            'std': quantile_returns.std(skipna=True, axis=1)
        }).T


class GroupedStudy(BaseEventReturnStudy):
    """
    Compute quantile cumulative returns separately for each group.
    Demean those returns accordingly to 'group_adjust' and 'demeaned'
    """
    def analyze(self, cumulative_return_func, factor_data, demean_by=None,
                group_adjust=None, demeaned=None):
        group_results = []
        for group, group_data in factor_data.groupby('group', observed=True):
            group_demean_by = self._get_demean_series(factor_data, group_data, group_adjust, demeaned)
            quantile_result = super().analyze(
                cumulative_return_func=cumulative_return_func,
                factor_data=group_data,
                demean_by=group_demean_by
            )
            if quantile_result.empty:
                continue
            quantile_result['group'] = group
            quantile_result.set_index('group', append=True, inplace=True)
            group_results.append(quantile_result)
        return pd.concat(group_results) if group_results else pd.DataFrame()

    @staticmethod
    def _get_demean_series(factor_data, group_data, group_adjust, demeaned):
        if group_adjust:
            return group_data['factor_quantile']
        elif demeaned:
            return factor_data['factor_quantile']
        return None


class GroupAdjustedStudy(BaseEventReturnStudy):
    """
    Demean group returns by group
    """
    def analyze(self, cumulative_return_func, factor_data, demean_by=None,
                group_adjust=None, demeaned=None):
        group_results = []
        for _, group_data in factor_data.groupby('group', observed=True):
            group_quantile = group_data['factor_quantile']
            quantile_returns = group_quantile.groupby(group_quantile).apply(
                cumulative_return_func,
                demean_by=group_quantile
            )
            group_results.append(quantile_returns)
        combined_returns = pd.concat(group_results, axis=1)
        return pd.DataFrame({
            'mean': combined_returns.mean(axis=1),
            'std': combined_returns.std(axis=1)
        }).unstack().stack(level=0, future_stack=True)


class DemeanedStudy(BaseEventReturnStudy):
    """
    Demean returns across the universe
    """
    def analyze(self, cumulative_return_func, factor_data, demean_by=None,
                group_adjust=None, demeaned=None):
        return super().analyze(
            cumulative_return_func=cumulative_return_func,
            factor_data=factor_data,
            demean_by=factor_data['factor_quantile']
        )


def compute_conditional_expectation_quantile(
        df: pd.DataFrame,
        x_col: str = "factor",
        y_col: str = "1D",
        group_col: str = "group",
        n_bins: int | None = None,
        n_quantiles: int | None = 20,
        q_clip: int = 0,
        binning: Literal["global", "cross_sectional", "grouped"] = "global",
        # agg_time: bool = True,
        agg_cross: bool = True,
        by_group: bool = False,
        asset: str | None = None,
) -> pd.DataFrame:
    """
    Compute conditional expectation E[y | x_bin] under aggregation modes.
    Returns a DataFrame with columns [group, x_mid, mean, sem] for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Should have columns corresponding to x_col, y_col, group_col, and (optionally) 'date' and 'asset'.
    x_col : str
        Name of the column to bin (the conditioning variable).
    y_col : str
        Name of the variable whose conditional mean is computed.
    group_col : str
        Name of the column indicating group labels.
    n_bins : Optional[int]
        Number of equal-width bins for x_col. Mutually exclusive with n_quantiles.
    n_quantiles : Optional[int]
        Number of quantile bins for x_col. Mutually exclusive with n_bins.
    q_clip : int
        Number of bins to exclude from each tail to reduce outlier influence.
    binning : Literal["global", "cross_sectional", "grouped"]
        Binning to use for x_col.
    agg_cross : bool
        If True, compute within-date bin means and then aggregate across time.
    by_group : bool
        If True, compute statistics separately for each group.
    asset : Optional[str]
        If set, select only this asset using the 'asset' index level.

    Returns
    -------
    pd.DataFrame
        DataFrame for plotting, with columns [group, x_mid, mean, sem].
    """

    if (n_bins is not None) and (n_quantiles is not None):
        raise ValueError("Specify only one of `n_bins` or `n_quantiles`.")
    if binning not in {"global", "cross_sectional", "grouped"}:
        raise ValueError("`binning` must be one of {'global', 'cross_sectional', 'grouped'}.")
    if asset is not None:
        if agg_cross:
            warnings.warn("agg_cross=True ignored for single asset mode.")
            agg_cross = False
        if binning != "global":
            warnings.warn("Switching to global binning for single asset mode.")
            binning = "global"

    def assign_bins(data: pd.DataFrame) -> pd.Series:
        if n_quantiles is not None:
            return pd.qcut(data[x_col], q=n_quantiles, duplicates='drop')
        elif n_bins is not None:
            return pd.cut(data[x_col], bins=n_bins)
        else:
            raise ValueError("Specify either `n_bins` or `n_quantiles`.")

    def compute_for_subset(sub_df: pd.DataFrame, label: str) -> pd.DataFrame:
        df_work = sub_df[[x_col, y_col, group_col]].dropna()

        # Ensure date index for CS aggregation
        if (agg_cross or binning == "cross_sectional") and 'date' not in df_work.index.names:
            if 'date' in df_work.columns:
                df_work = df_work.set_index('date')
            else:
                raise ValueError("Missing 'date' column for cross-sectional processing.")

        # Assign bins
        if binning == "cross_sectional":
            df_work['bin'] = df_work.groupby('date', group_keys=False, observed=True).apply(assign_bins)
        elif binning == "grouped":
            if group_col not in df_work.columns:
                raise ValueError(f"Missing column '{group_col}' for grouped binning.")
            df_work['bin'] = df_work.groupby(group_col, group_keys=False, observed=True).apply(assign_bins)
        else:  # global
            df_work['bin'] = assign_bins(df_work)

        # Cross-sectional (e.g. by date) aggregation
        if agg_cross:
            # Group by date, then bin within each date
            df_work = (
                df_work.groupby('date')
                .apply(lambda g: g.groupby('bin', observed=True)[y_col].mean())
                .reset_index(level=0, drop=True)
                .to_frame(name=y_col)
            )

        stats = (
            df_work.groupby('bin', observed=True)[y_col]
            .agg(['mean', 'std', 'count'])
            .assign(sem=lambda x: x['std'] / np.sqrt(x['count']))
        )

        stats['x_mid'] = stats.index.map(lambda b: b.mid)
        stats['group'] = label
        stats = stats.reset_index(drop=True)

        if q_clip > 0:
            stats = stats.iloc[q_clip:-q_clip]
        return stats[['group', 'x_mid', 'mean', 'sem']]

    # Asset-specific mode
    if asset is not None:
        df_asset = df.xs(asset, level='asset', drop_level=False)
        return compute_for_subset(df_asset, label=asset)

    # Grouped mode
    if by_group:
        return (
            df.groupby(group_col, dropna=True)
            .apply(lambda g: compute_for_subset(g, label=g.name))
            .reset_index(drop=True)
        )
    else:
        return compute_for_subset(df, label="all")

