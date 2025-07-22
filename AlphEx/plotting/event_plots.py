from typing import List

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes
import seaborn as sns

from AlphEx.plotting import DECIMAL_TO_BPS


def plot_events_distribution(events: pd.Series, num_bars: int = 50, ax: List[Axes] | None = None):
    """
    Plots the distribution of events in time.

    Parameters
    ----------
    events : pd.Series
        A pd.Series whose index contains at least 'date' level.
    num_bars : integer, optional
        Number of bars to plot
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    start = events.index.get_level_values('date').min()
    end = events.index.get_level_values('date').max()
    group_interval = (end - start) / num_bars
    grouper = pd.Grouper(level='date', freq=int(group_interval))
    events.groupby(grouper).count().plot(kind="bar", grid=False, ax=ax)
    ax.set(ylabel='Number of events',
           title='Distribution of events in time',
           xlabel='Date')

    return ax


def plot_quantile_average_cumulative_return(avg_cumulative_returns: pd.DataFrame,
                                            by_quantile: bool = False,
                                            std_bar: bool = False,
                                            title: str | None = None,
                                            ax: List[Axes] | None = None):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    avg_cumulative_returns: pd.Dataframe
        The format is the one returned by
        AlphEx.analytics.event_study.average_cumulative_return_by_quantile
    by_quantile : boolean, optional
        Disaggregated figures by quantile (useful to clearly see std dev bars)
    std_bar : boolean, optional
        Plot standard deviation plot
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    avg_cumulative_returns = avg_cumulative_returns.multiply(DECIMAL_TO_BPS)
    quantiles = len(avg_cumulative_returns.index.levels[0].unique())
    palette = [cm.coolwarm(i) for i in np.linspace(0, 1, quantiles)]
    palette = palette[::-1]  # we want negative quantiles as 'red'

    if by_quantile:

        if ax is None:
            v_spaces = ((quantiles - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False, sharey=False, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for i, (quantile, q_ret) in enumerate(avg_cumulative_returns.groupby(level='factor_quantile')):
            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = 'Quantile ' + str(quantile)
            mean.plot(ax=ax[i], color=palette[i])
            ax[i].set_ylabel('Mean Return (bps)')

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax[i].errorbar(std.index, mean, yerr=std,
                               fmt='none', ecolor=palette[i], label='none')

            ax[i].axvline(x=0, color='k', linestyle='--')
            ax[i].legend()
            i += 1

    else:

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        for i, (quantile, q_ret) in enumerate(avg_cumulative_returns.groupby(level='factor_quantile')):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = 'Quantile ' + str(quantile)
            mean.plot(ax=ax, color=palette[i])

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax.errorbar(std.index, mean, yerr=std,
                            fmt='none', ecolor=palette[i], label='none')
            i += 1

        ax.axvline(x=0, color='k', linestyle='--')
        ax.legend()
        ax.set(ylabel="Mean Return (bps)",
               title='Average Cumulative Returns by Quantile' if title is None else title,
               xlabel="Periods")

    return ax


def plot_conditional_expectation_quantile(
    conditional_expectation_df: pd.DataFrame,
    sns_reg: bool = False,
    ax: List[Axes] | None = None
) -> List[Axes]:
    """
    Plot conditional expectation curves.

    Parameters
    ----------
    conditional_expectation_df : pd.DataFrame
        DataFrame output from compute_conditional_expectation_quantile.
    sns_reg : bool, default False
        Use seaborn.regplot if True, else error-bar line plots.
    ax : Optional[List[Axes]]
        List of matplotlib Axes to plot into. If None, new subplots are created.

    Returns
    -------
    List[Axes]
        List of matplotlib Axes containing the plots.
    """

    # Flexible column detection
    colmap = {c.lower(): c for c in conditional_expectation_df.columns}
    x_col = colmap.get("x_mid")
    mean_col = colmap.get("mean")
    sem_col = colmap.get("sem")
    group_col = colmap.get("group")

    if x_col is None or mean_col is None:
        raise ValueError("DataFrame must contain 'x_mid' and 'mean' columns.")

    # Identify groups to plot
    if group_col is not None:
        groups = conditional_expectation_df[group_col].dropna().unique()
    else:
        groups = [None]

    n_groups = len(groups)

    # Create subplots if axes not provided
    if ax is None:
        fig, axs = plt.subplots(1, n_groups, figsize=(6 * n_groups, 4), squeeze=False)
        axs = axs.flatten()
    else:
        axs = ax

    # Plot each group separately
    for i, group in enumerate(groups):
        ax_i = axs[i]
        df_g = (
            conditional_expectation_df
            if group is None
            else conditional_expectation_df[conditional_expectation_df[group_col] == group]
        )

        if sns_reg:
            sns.regplot(
                x=x_col,
                y=mean_col,
                data=df_g,
                ax=ax_i,
                scatter_kws={"s": 25},
                line_kws={"color": "C0"}
            )
        else:
            yerr = df_g[sem_col] if sem_col else None
            ax_i.errorbar(
                df_g[x_col],
                df_g[mean_col],
                yerr=yerr,
                fmt="-o",
                capsize=3,
                color="C0"
            )

        ax_i.set_xlabel("Binned Factor Value (Midpoint)")
        ax_i.set_ylabel("Conditional Expectation")
        ax_i.grid(True)

        if group is not None:
            ax_i.set_title(str(group))

    axs[0].figure.suptitle("Conditional Expectation Quantile Plot", fontsize=14)
    axs[0].figure.tight_layout(rect=[0, 0, 1, 0.95])

    return list(axs)


