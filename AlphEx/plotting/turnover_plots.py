from typing import List

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_factor_rank_auto_correlation(
        factor_autocorrelation: pd.Series,
        period: int = 1,
        ax: List[Axes] | None = None
):
    """
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
    period: int, optional
        Period over which the autocorrelation is calculated
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_autocorrelation.plot(title=f'{period}D Period Factor Rank Autocorrelation', ax=ax)
    ax.set(ylabel='Autocorrelation Coefficient', xlabel='')
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    ax.text(.05, .95, f"Mean {factor_autocorrelation.mean():.3f}",
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    return ax


def plot_top_bottom_quantile_turnover(quantile_turnover: pd.DataFrame, period: int = 1, ax: List[Axes] | None = None):
    """
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame column a quantile).
    period: int, optional
        Period over which to calculate the turnover.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame({
        'top quantile turnover': quantile_turnover[max_quantile],
        'bottom quantile turnover': quantile_turnover[min_quantile],
    })
    turnover.plot(title=f"{period}D Period Top and Bottom Quantile Turnover", ax=ax, alpha=0.6, lw=0.8)
    ax.set(ylabel="Proportion Of Names New To Quantile", xlabel="")

    return ax
