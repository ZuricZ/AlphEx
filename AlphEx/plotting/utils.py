from functools import wraps
from typing import Callable, Dict, Literal

import seaborn as sns


def customize(func: Callable) -> Callable:
    """
    Decorator to set plotting context and axes style during function call.
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            color_palette = sns.color_palette('colorblind')
            with plotting_context(), axes_style(), color_palette:
                sns.despine(left=True)
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def plotting_context(
        context: Literal["paper", "notebook", "talk", "poster"] = 'notebook',
        font_scale: float = 1.5, rc: Dict | None = None
):
    """
    Create alphalens default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.plotting_context(font_scale=2):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(
        style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = 'darkgrid',
        rc: Dict | None = None
):
    """Create alphalens default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.axes_style(style='whitegrid'):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)