from typing import List

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import rv_continuous
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes



def plot_ic_timeseries(ic: pd.DataFrame, ax: Axes | None = None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.items()):
        ic.plot(alpha=0.7, ax=a, lw=0.7, color='steelblue')
        ic.rolling(window=22).mean().plot(
            ax=a,
            color='forestgreen',
            lw=2,
            alpha=0.8
        )

        a.set(ylabel='IC', xlabel="")
        a.set_title(f"{period_num} Period Forward Return Information Coefficient (IC)")
        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.legend(['IC', '1 month moving avg'], loc='upper right')
        a.text(.05, .95, f"Mean {ic.mean():.3f} \n Std. {ic.std():.3f}",
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])

    return ax


def plot_ic_hist(ic: pd.DataFrame, ax: List[Axes] | None = None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.items()):
        sns.histplot(ic.fillna(0), kde=True, stat='density', ax=a)
        a.set(title=f"{period_num} Period IC", xlabel='IC')
        a.set_xlim([-1, 1])
        a.text(.05, .95, f"Mean {ic.mean():.3f} \n Std. {ic.std():.3f}",
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def plot_ic_qq(
        ic: pd.DataFrame, theoretical_dist: rv_continuous = stats.norm, ax: List[Axes] | None = None
):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats.rv_continuous
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = 'Theoretical'

    for a, (period_num, ic) in zip(ax, ic.items()):
        sm.qqplot(ic.fillna(0).to_numpy(), theoretical_dist, fit=True,
                  line='45', ax=a)
        a.set(title=f"{period_num} Period IC {dist_name} Dist. Q-Q",
              ylabel='Observed Quantile',
              xlabel=f'{dist_name} Distribution Quantile')

    return ax


def plot_ic_by_group(ic_group: pd.DataFrame, ax: List[Axes] | None = None):
    """
    Plots Spearman Rank Information Coefficient for a given factor over
    provided forward returns. Separates by group.

    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    ic_group.plot(kind='bar', ax=ax)

    ax.set(title="Information Coefficient By Group", xlabel="")
    ax.set_xticklabels(ic_group.index, rotation=45, ha='right')

    return ax


def plot_monthly_ic_heatmap(mean_monthly_ic, ax: List[Axes] | None = None):
    """
    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_monthly_ic = mean_monthly_ic.copy()

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["year", "month"])

    for a, (periods_num, ic) in zip(ax, mean_monthly_ic.items()):

        sns.heatmap(
            ic.unstack(),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.coolwarm_r,
            cbar=False,
            ax=a)
        a.set(ylabel='', xlabel='')

        a.set_title(f"Monthly Mean {periods_num} Period IC")

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax