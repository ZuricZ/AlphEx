import pandas as pd
from scipy import stats

from IPython.display import display

from AlphEx.plotting import DECIMAL_TO_BPS


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def plot_returns_table(
        alpha_beta: pd.DataFrame, mean_return_quantile: pd.DataFrame, mean_return_spread_quantile: pd.Series
):
    returns_table = alpha_beta.copy()
    returns_table.loc["Time-wise mean Return Top Quantile (bps)"] = mean_return_quantile.iloc[-1] * DECIMAL_TO_BPS
    returns_table.loc["Time-wise mean Return Bottom Quantile (bps)"] = mean_return_quantile.iloc[0] * DECIMAL_TO_BPS
    returns_table.loc["Time-wise mean Spread (bps)"] = mean_return_spread_quantile.mean() * DECIMAL_TO_BPS

    print("Returns Analysis")
    print_table(returns_table.apply(lambda x: x.round(3)))


def plot_turnover_table(autocorrelation_data, quantile_turnover):
    turnover_dict = {}
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].items():
            row_key = f"Quantile {quantile} Mean Turnover "
            col_key = f"{period}D"
            turnover_dict.setdefault(row_key, {})[col_key] = p_data.mean()

    turnover_table = pd.DataFrame.from_dict(turnover_dict, orient='index')

    auto_corr_dict = {
        f"{period}D": p_data.mean()
        for period, p_data in autocorrelation_data.items()
    }

    auto_corr = pd.DataFrame(
        [auto_corr_dict],
        index=["Mean Factor Rank Autocorrelation"]
    )

    print("Turnover Analysis")
    print_table(turnover_table.apply(lambda x: x.round(3)))
    print_table(auto_corr.apply(lambda x: x.round(3)))


def plot_information_table(ic_data: pd.DataFrame | pd.Series):
    ic_summary = {
        "IC Mean": ic_data.mean(),
        "IC Std.": ic_data.std(),
        "Risk-Adjusted IC": ic_data.mean() / ic_data.std(),
        "t-stat(IC)": stats.ttest_1samp(ic_data, 0).statistic,
        "p-value(IC)": stats.ttest_1samp(ic_data, 0).pvalue,
        "IC Skew": stats.skew(ic_data),
        "IC Kurtosis": stats.kurtosis(ic_data),
    }

    ic_summary_table = pd.DataFrame(ic_summary)
    if isinstance(ic_data, pd.Series):
        ic_summary_table = ic_summary_table.to_frame().T
    print("Information Analysis")
    print_table(ic_summary_table.apply(lambda x: x.round(3)).T)


def plot_quantile_statistics_table(factor_data: pd.DataFrame):
    quantile_stats = factor_data.groupby('factor_quantile').agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] / quantile_stats['count'].sum() * 100.

    print("Quantiles Statistics")
    print_table(quantile_stats)