from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

from AlphEx.plotting import DECIMAL_TO_BPS
from AlphEx.analytics.return_calculation import compute_cumulative_returns


def plot_quantile_returns_bar(
        mean_ret_by_q: pd.DataFrame,
        by_group: bool = False,
        ylim_percentiles: Tuple = None,
        ax: List[Axes] | None = None
):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_ret_by_q = mean_ret_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.to_numpy(), ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.to_numpy(), ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if by_group:
        num_group = len(
            mean_ret_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False, sharey=True, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level='group')):
            (cor.xs(sc, level='group')
                .multiply(DECIMAL_TO_BPS)
                .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Mean Return (bps)',
                  ylim=(ymin, ymax))

        if num_group < len(ax):
            ax[-1].set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        mean_ret_by_q.multiply(DECIMAL_TO_BPS).plot(
            kind='bar', title="Time-wise Return By Factor Quantile",
            ax=ax
        )
        ax.set(xlabel='', ylabel='Mean Return (bps)', ylim=(ymin, ymax))

        return ax


def plot_quantile_returns_violin(
        return_by_q: pd.DataFrame,
        ylim_percentiles: Tuple | None = None,
        ax: List[Axes] | None = None
):
    """
    Plots a violin box plot of period wise returns for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.to_numpy(), ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.to_numpy(), ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = return_by_q.multiply(DECIMAL_TO_BPS)
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(data=unstacked_dr,
                   x='factor_quantile',
                   hue='forward_periods',
                   y='return',
                   orient='v',
                   cut=0,
                   inner='quartile',
                   ax=ax)
    ax.set(xlabel='', ylabel='Return (bps)',
           title="Period Wise Return By Factor Quantile",
           ylim=(ymin, ymax))

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax


def plot_mean_quantile_returns_spread_time_series(
        mean_returns_spread: pd.Series,
        std_err: pd.Series | None = None,
        bandwidth: float = 1.,
        ax: List[Axes] | None = None
):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by period.
    std_err : pd.Series
        Series with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if isinstance(mean_returns_spread, pd.DataFrame):
        if ax is None:
            ax = [None for a in mean_returns_spread.columns]

        ymin, ymax = (None, None)
        for (i, a), (name, fr_column) in zip(enumerate(ax), mean_returns_spread.items()):
            stdn = None if std_err is None else std_err[name]
            a = plot_mean_quantile_returns_spread_time_series(fr_column,
                                                              std_err=stdn,
                                                              ax=a)
            ax[i] = a
            curr_ymin, curr_ymax = a.get_ylim()
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        for a in ax:
            a.set_ylim([ymin, ymax])

        return ax

    if mean_returns_spread.isnull().all():
        return ax

    periods = mean_returns_spread.name
    title = f"Top Minus Bottom Quantile Mean Return ({periods if periods is not None else ''} Period Forward Return)"

    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))

    mean_returns_spread_bps = mean_returns_spread * DECIMAL_TO_BPS

    mean_returns_spread_bps.plot(alpha=0.4, ax=ax, lw=0.7, color='forestgreen')
    mean_returns_spread_bps.rolling(window=22).mean().plot(
        color='orangered',
        alpha=0.7,
        ax=ax
    )
    ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')

    if std_err is not None:
        std_err_bps = std_err * DECIMAL_TO_BPS
        upper = mean_returns_spread_bps.to_numpy() + (std_err_bps * bandwidth)
        lower = mean_returns_spread_bps.to_numpy() - (std_err_bps * bandwidth)
        ax.fill_between(mean_returns_spread.index,
                        lower,
                        upper,
                        alpha=0.3,
                        color='steelblue')

    ylim = np.nanpercentile(abs(mean_returns_spread_bps.to_numpy()), 95)
    ax.set(ylabel='Difference In Quantile Mean Return (bps)',
           xlabel='',
           title=title,
           ylim=(-ylim, ylim))
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    return ax

def plot_cumulative_returns(factor_returns: pd.Series,
                            period: pd.Timedelta | str,
                            freq: pd.DateOffset | None = None,
                            title: str | None = None,
                            ax: List[Axes] | None = None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_returns = compute_cumulative_returns(factor_returns)

    factor_returns.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(ylabel='Cumulative Returns',
           title=f"Portfolio Cumulative Return ({period} Fwd Period)" if title is None else title,
           xlabel='')
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


def plot_cumulative_returns_by_quantile(quantile_returns: pd.DataFrame,
                                        period: pd.Timedelta | str,
                                        freq: pd.DateOffset | None = None,
                                        ax: List[Axes] | None = None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    quantile_returns : pd.DataFrame
        Returns by factor quantile
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    ret_wide = quantile_returns.unstack('factor_quantile')

    cum_ret = ret_wide.apply(compute_cumulative_returns)

    cum_ret = cum_ret.loc[:, ::-1]  # we want negative quantiles as 'red'

    cum_ret.plot(lw=2, ax=ax, cmap=cm.coolwarm)
    ax.legend()
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set(ylabel="Log Cumulative Returns",
           title=f'''Cumulative Return by Quantile
                        ({period} Period Forward Return)''',
           xlabel="",
           yscale="symlog",
           yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax

