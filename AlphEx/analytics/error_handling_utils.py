import functools


class NonMatchingTimezoneError(Exception):
    pass


class MaxDataLossExceededError(Exception):
    pass


def rethrow(exception: Exception, additional_message: str) -> None:
    """
    Re-raise an exception with additional context, preserving the traceback.

    Parameters
    ----------
    exception : Exception
        The original exception to re-raise.
    additional_message : str
        Extra context to include in the error message.
    """
    new_message = f"{exception} — {additional_message}"
    raise type(exception)(new_message) from exception


def handle_nonunique_bin_edges(func):
    """
    Decorator that provides a more informative error message when calculation
    of bins/quantiles fails due to non-unique bin edges. This error often occurs
    when the input contains many identical values that span multiple quantiles.
    """
    message = """
    
        An error occurred while computing bins/quantiles on the input provided.
        This usually happens when the input contains too many identical
        values and they span more than one quantile. The quantiles are chosen
        to have the same number of records each, but the same value cannot span
        multiple quantiles. Possible workarounds are:
        1 - Decrease the number of quantiles
        2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
            number of records per quantile
        3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
            buckets to be evenly spaced according to the values themselves, while
            'quantiles' forces the buckets to have the same number of records.
        4 - for factors with discrete values use the 'bins' option with custom
            ranges and create a range for each discrete value
        Please see AlphEx.analytics.factor_processing.factor_analysis_pipe documentation for
        full documentation of 'bins' and 'quantiles' options.
    
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise

    return wrapper