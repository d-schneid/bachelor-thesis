import timeit
import numpy as np
import pandas as pd

from approximation.paa import PAA


def generate_standard_normal_ts(ts_size):
    """
    Generate a time series where each point is randomly drawn from the standard
    normal distribution.

    :param ts_size: int
        The number of points the generated time series shall have.
    :return: dataframe of shape (ts_size, 1)
    """

    return pd.DataFrame(np.random.normal(scale=1, size=ts_size))


def measure_paa_approximation(df_ts, window_size, num_repetitions=1):
    """
    Measure the execution time of the PAA approximation of the given time
    series.

    :param df_ts: dataframe of shape (ts_size, num_ts)
        The time series that shall be transformed into their PAA
        representations while measuring the execution time.
    :param window_size: int
        The size of the window that shall be used for transforming the given
        time series into their PAA representations.
    :param num_repetitions: int
        The number of repetitions the given time series shall be transformed
        into their PAA representations.
    :return: float
        The execution time in seconds.
    """

    paa = PAA(window_size)
    execution_time = timeit.timeit(lambda: paa.transform(df_ts), number=num_repetitions)

    return execution_time


def measure_symbolic_discretization(df_paa, df_ts, window_size, sax_variant, num_repetitions=1):
    """
    Measure the execution time of the symbolic transformation of the given time
    series.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations of the given time series.
    :param df_ts: dataframe of shape (ts_size, num_ts)
        The time series that shall be transformed into their symbolic
        representations while measuring the execution time.
    :param window_size: int
        The size of the window that was used to create the given PAA
        representations.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the given time series
        into their symbolic representations.
    :param num_repetitions: int
        The number of repetitions the given time series shall be transformed
        into their symbolic representations.
    :return: float
        The execution time in seconds.
    """

    execution_time = timeit.timeit(lambda: sax_variant.transform(df_paa=df_paa, df_norm=df_ts,
                                                                 window_size=window_size), number=num_repetitions)
    return execution_time
