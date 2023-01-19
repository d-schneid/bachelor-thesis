import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore


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


def constant_segmentation_overlapping(ts_size, len_subsequence, gap=1):
    """
    Compute the segment boundaries for the extraction of overlapping
    subsequences from original time series.

    :param ts_size: int
        The size of the time series from that subsequences shall be extracted.
    :param len_subsequence: int
        The length of the subsequences that shall be extracted from a time
        series.
    :param gap: int (default = 1)
        The gap between two consecutive subsequences within the corresponding
        time series when extracting them.
    :return:
        start: np.array of shape (num_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end: np.array of shape (num_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    """

    if not len_subsequence <= ts_size:
        raise ValueError("'len_subsequence' must be lower than or equal to "
                         "'ts_size' ({0} > {1})."
                         "".format(len_subsequence, ts_size))

    start = np.arange(0, ts_size - len_subsequence + 1, gap)
    end = np.arange(len_subsequence, ts_size + 1, gap)
    return start, end


def load_parquet_to_df_list(path):
    """
    Load the data of Parquet files into individual dataframes.
    If the given path belongs to a file, the data of this file are loaded.
    If the given path belongs to a directory, the data of all files located in
    this directory are loaded.

    :param path: string
        Path to the Parquet file or directory to be read. Relative to the
        directory "0_data".
    :return:
        list of len(list) = num_loaded_files containing dataframes of shape
        (ts_size, 1 + num_metadata_cols)
    :raises:
        ValueError: If a dataframe containing a time series, potentially with
                    its metadata, contains NaN. Either the respective time
                    series or its metadata contain NaN or the respective time
                    series together with its metadata do not have the same
                    length.
        Exception: On any failure, e.g. if the given path does not exist or
                   the files are corrupted.
    """

    error_msg = "Dataframe containing a time series, potentially with its " \
                "metadata, contains NaN. Either the respective time series " \
                "or its metadata contain NaN or the respective time series " \
                "together with its metadata do not have the same length."

    path = Path(path)

    if os.path.isdir(path):
        df_list = []
        for parquet_file in path.glob("*parquet"):
            df = pd.read_parquet(parquet_file, engine="fastparquet")
            if df.isnull().values.any():
                raise ValueError(error_msg)
            df_list.append(df)
        return df_list

    df = pd.read_parquet(path, engine="fastparquet")
    if df.isnull().values.any():
        raise ValueError(error_msg)
    return [df]


def z_normalize(df_ts):
    """
    Z-normalize the given time series dataset. All time series in the dataset
    are z-normalized individually.

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
