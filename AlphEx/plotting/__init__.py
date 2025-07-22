# Shared constant available at package level
DECIMAL_TO_BPS = 10000

# Import and expose key functions from submodules
from .tables import (
    plot_returns_table,
    plot_turnover_table,
    plot_information_table,
    plot_quantile_statistics_table,
)

from .ic_plots import (
    plot_ic_timeseries,
    plot_ic_hist,
    plot_ic_qq,
    plot_ic_by_group,
    plot_monthly_ic_heatmap,
)

from .return_plots import (
    plot_quantile_returns_bar,
    plot_quantile_returns_violin,
    plot_mean_quantile_returns_spread_time_series,
    plot_cumulative_returns,
    plot_cumulative_returns_by_quantile,
)

from .turnover_plots import (
    plot_factor_rank_auto_correlation,
    plot_top_bottom_quantile_turnover,
)


from .event_plots import (
    plot_events_distribution, plot_quantile_average_cumulative_return,
)

from .utils import (
    customize,
    plotting_context,
    axes_style,
)

# Define __all__ for explicit "import *" behavior
__all__ = [
    "DECIMAL_TO_BPS",
    # Tables
    "plot_returns_table",
    "plot_turnover_table",
    "plot_information_table",
    "plot_quantile_statistics_table",
    # IC Plots
    "plot_ic_timeseries",
    "plot_ic_hist",
    "plot_ic_qq",
    "plot_ic_by_group",
    "plot_monthly_ic_heatmap",
    # Return Plots
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_mean_quantile_returns_spread_time_series",
    "plot_quantile_average_cumulative_return",
    "plot_cumulative_returns",
    "plot_cumulative_returns_by_quantile",
    # Turnover Plots
    "plot_factor_rank_auto_correlation",
    "plot_top_bottom_quantile_turnover",
    # Event Plots
    "plot_events_distribution",
    # Utilities
    "customize",
    "plotting_context",
    "axes_style",
]
