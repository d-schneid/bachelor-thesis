import numpy as np
import pandas as pd


def _discretize(df_norm, df_breakpoints):
    """
    Discretize time series points into bins based on the given breakpoints.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series that shall be discretized.
    :param df_breakpoints: dataframe of shape (num_breakpoings, num_ts)
        The indiviual breakpoints for each given time series.
    :return: dataframe of shape (ts_size, num_ts)
        Each point got assigned its corresponding bin id based on the given
        breakpoints.
    """

    discretized = []
    bin_ids = np.array([i for i in range(df_breakpoints.shape[0] + 1)])
    for i in range(df_norm.shape[1]):
        current_ts = df_norm.iloc[:, i]
        current_breakpoints = df_breakpoints.iloc[:, i]
        bin_idxs = np.searchsorted(current_breakpoints, current_ts, side="right")
        df_discretized = pd.DataFrame(data=bin_ids[bin_idxs], columns=[i])
        discretized.append(df_discretized)

    return pd.concat(discretized, axis=1)
