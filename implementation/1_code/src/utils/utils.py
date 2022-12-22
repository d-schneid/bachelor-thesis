import numpy as np


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
