import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore


# so that all data paths can be given relative to the directory "0_data"
DATA_DIR = "../../0_data"


def constant_segmentation(ts_size, window_size):
    """
    Compute the segment boundaries for PAA.

    :param ts_size: int
        The size of the time series (number of points).
    :param window_size: int
        The size of the segmentation window.
    :return:
        start : np.array
            The index of the lower segment bound (inclusive) for each window.
        end : np.array
            The index of the upper segment bound (exclusive) for each window.
        num_segments : int
            The number of segments.

    Examples
    --------
    start, end, num_segments = segmentation(ts_size=11, window_size=3)
    print(start)
    [0, 3, 6, 9]
    print(end)
    [3, 6, 9, 11]
    print(num_segments)
    4
    """

    if not isinstance(ts_size, (int, np.integer)):
        raise TypeError("'ts_size' must be an integer.")
    if not ts_size >= 1:
        raise ValueError("'ts_size' must be an integer greater than or equal "
                         "to 1 (got {0}).".format(ts_size))
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("'window_size' must be an integer.")
    if not window_size >= 1:
        raise ValueError("'window_size' must be an integer greater than or "
                         "equal to 1 (got {0}).".format(window_size))
    if not window_size <= ts_size:
        raise ValueError("'window_size' must be lower than or equal to "
                         "'ts_size' ({0} > {1}).".format(window_size, ts_size))

    bounds = np.arange(0, ts_size + 1, window_size)
    remainder = ts_size % window_size
    # Do not cut off last points if it is a significant number
    if remainder > window_size / 2:
        bounds = np.append(bounds, ts_size)
    start = bounds[:-1]
    end = bounds[1:]
    num_segments = start.size
    return start, end, num_segments


def load_parquet_to_df(path):
    """
    Load Parquet file for univariate and multivariate time series dataset.
    If the given path belongs to a file, the data of this file are loaded.
    If the given path belongs to a directory, the data of all files located in
    this directory are loaded.

    :param path: string
        Path to the Parquet file or directory to be read. Relative to the
        directory "0_data".
    :return:
        dataframe of shape (ts_size, num_ts)
    :raises:
        ValueError: If the time series dataset contains NaN. Either some time
                    series of the dataset contain NaN or some time series of
                    the dataset do not have the same size.
        Exception: On any failure, e.g. if the given path does not exist or
                   the files are corrupted.
    """

    error_msg = "Time series dataset contains NaN. Either some time series " \
                "of the dataset contain NaN or some time series of the " \
                "dataset do not have the same size."

    path = Path(os.path.join(DATA_DIR, path))

    if os.path.isdir(path):
        df = pd.concat((pd.read_parquet(parquet_file, engine="fastparquet")
                        for parquet_file in path.glob("*.parquet")), axis=1)
        if df.isnull().values.any():
            raise ValueError(error_msg)
        return df

    df = pd.read_parquet(path, engine="fastparquet")
    if df.isnull().values.any():
        raise ValueError(error_msg)

    return df


def z_normalize(df_ts):
    """
    Z-normalize the given time series dataset.

    :param df_ts: dataframe of shape (ts_size, num_ts)
        The time series dataset that shall be z-normalized.
    :return:
        dataframe of shape (ts_size, num_ts)
    :raises:
        ValueError: If at least one time series consists of points that all
        have the same value.
    """

    fst_row = df_ts.values[0]
    if True in (fst_row == df_ts.values).all(axis=0):
        raise ValueError("Z-normalization does not work, because at least one"
                         "time series consists of points that all have the"
                         "same value.")
    return zscore(df_ts)


def interpolate_segments(df_segments, ts_size, window_size):
    """
    Approximate the original time series dataset by interpolating the segmented
    representations into a time series dataset with the same size by assigning
    each point the value of its segment.

    :param df_segments: dataframe of shape (num_segments, num_ts)
        The segmented representation of the time series dataset.
    :param ts_size: int
        The size of the original time series.
    :param window_size: int
        The size of the segments with which the given segmented representations
        were created.
    :return:
        dataframe of shape (ts_size, num_ts)
    """

    start, end, num_segments = constant_segmentation(ts_size, window_size)
    df_interpolate = pd.DataFrame(index=range(ts_size), columns=df_segments.columns)
    for i in range(num_segments):
        df_interpolate.iloc[start[i]:end[i]] = df_segments.iloc[i]

    remainder = ts_size % window_size
    # remaining points at the end get the value of the last segment
    if 0 < remainder <= window_size / 2:
        df_interpolate.iloc[-remainder:] = df_segments.iloc[-1]

    return df_interpolate
