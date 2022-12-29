import numpy as np
import pandas as pd

from utils.utils import constant_segmentation, interpolate_segments


class PAA:
    """
    Piecewise Aggregate Approximation (PAA).

    :param window_size: int (default = 1)
        Length of the window to segment the time series.

    Examples
    --------
    paa = PAA(window_size=2)
    df_ts = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9],
                      [10,11,12]])
    df_paa = paa.transform(df_ts)
    print(df_paa)
    [[2.5, 3.5, 4.5],
     [8.5, 9.5, 10.5]]

    References
    ----------
    [1] Keogh, E., Chakrabarti, K., Pazzani, M., & Mehrotra, S. (2001).
    Dimensionality reduction for fast similarity search in large time series
    databases. Knowledge and information Systems, 3(3), 263-286.
    """

    def __init__(self, window_size=1):
        if window_size < 1:
            raise ValueError("The size of the window must be greater than 0")
        self.window_size = window_size

    def transform(self, df_ts):
        """
        Reduce the dimensionality of each time series by transforming it into
        its PAA representation.
        It does not modify the given time series dataset before computation,
        such as normalization. Therefore, the modification of the time series
        dataset (e.g. normalization) is the responsibility of the user.

        :param df_ts: dataframe of shape (ts_size, num_ts)
            The time series dataset.
        :return:
            dataframe of shape (num_segments, num_ts)
        """

        ts_size = df_ts.shape[0]
        start, end, num_segments = constant_segmentation(ts_size, self.window_size)
        segment_means = []

        for i in range(num_segments):
            segment = df_ts.iloc[start[i]:end[i]]
            segment_means.append(segment.mean(axis=0))

        df_paa = pd.DataFrame(data=np.array(segment_means),
                              index=range(num_segments),
                              columns=df_ts.columns)
        return df_paa

    def inv_transform(self, df_paa, ts_size):
        """
        Approximate the original time series dataset by transforming its
        PAA representations into a time series dataset with the same size
        by assigning each point the PAA value of its segment.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representation of the time series dataset.
        :param ts_size: int
            The size of the original time series.
        :return:
            dataframe of shape (ts_size, num_ts)
        """

        return interpolate_segments(df_paa, ts_size, self.window_size)
