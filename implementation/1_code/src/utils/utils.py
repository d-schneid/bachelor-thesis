import os
import pandas as pd
import numpy as np
from pathlib import Path
from tslearn.utils.utils import\
    (to_time_series as tslearn_to_time_series,
     to_time_series_dataset as tslearn_to_time_series_dataset)


DATA_DIR = "../../0_data"


def constant_segmentation(ts_size, window_size):
    """
    Compute the segment boundaries for PAA.

    :param ts_size: int
        The size of the time series (number of points).
    :param window_size: int
        The size of the segmentation window.
    :return:
        start : array
            The index of the lower segment bound (inclusive) for each window.
        end : array
            The index of the upper segment bound (exclusive) for each window.
        num_segments : int
            The number of segments.

    Examples
    --------
    start, end, num_segments = segmentation(ts_size=11, window_size=3)
    print(start)
    [0 3 6 9]
    print(end)
    [ 3  6  9 11]
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
    # Do not cut off last points if it is a significant number
    if ts_size % window_size > window_size / 2:
        bounds = np.append(bounds, ts_size)
    start = bounds[:-1]
    end = bounds[1:]
    num_segments = start.size
    return start, end, num_segments


# TODO: Error handling (e.g. if path does not exist)
# TODO: comments on expected datatypse, input, output
def load_parquet_to_df(path):
    path = Path(os.path.join(DATA_DIR, path))

    df = pd.DataFrame()

    if os.path.isdir(path):
        df = pd.concat((pd.read_parquet(parquet_file, engine="fastparquet")
                        for parquet_file in path.glob("*.parquet")), axis=1)
    elif os.path.isfile(path):
        df = pd.read_parquet(path, engine="fastparquet")

    return df


# TODO: comment and error handling
def from_df(df, dim=1):
    # dim is number of dimensions per time series, all equal dimensions
    # default are univariate time series
    quotient, remainder = divmod(len(df.columns), dim)
    if remainder != 0:
        raise ValueError("number of columns must be divisable by 'dim'"
                         "'({0} % {1} != 0).".format(len(df.columns), dim))

    ts_dataset = []
    for i in range(0, len(df.columns), dim):
        ts = df.iloc[:, i:i+dim].to_numpy()
        ts_dataset.append(ts)

    return np.stack(ts_dataset)


def to_df(X):
    df_set = []
    num_ts = X.shape[0]
    for i_ts in range(num_ts):
        df_set.append(pd.DataFrame(X[i_ts]))
    return pd.concat(df_set, axis=1)
