import numpy as np
import pandas as pd

from utils.utils import constant_segmentation


class PAA:
    # TODO: inherit from something like BaseApproximator

    def __init__(self, window_size=1):
        self.window_size = window_size
        # TODO: super parent class

    def transform(self, df_ts):
        """
        Reduce the dimensionality of each time series by transforming it into
        its PAA representation.

        :param df_ts: dataframe of shape (ts_size, num_ts)
            The time series dataset.
        :return:
            dataframe of shape (num_segments, num_ts)
        """

        ts_size = df_ts.shape[0]
        start, end, num_segments = constant_segmentation(ts_size, self.window_size)
        remainder = ts_size % self.window_size
        df_copy = df_ts
        if remainder <= self.window_size / 2:
            df_copy = df_ts.drop(df_ts.tail(remainder).index)

        segment_means = []
        for segment in np.array_split(df_copy, num_segments):
            segment_means.append(segment.mean(axis=0))

        df_paa = pd.DataFrame(data=np.array(segment_means), columns=df_ts.columns)
        return df_paa

    def inverse_transform(self, df_paa, ts_size):
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

        start, end, num_segments = constant_segmentation(ts_size, self.window_size)
        df_inv = pd.DataFrame(columns=df_paa.columns, index=range(ts_size))
        for i in range(df_paa.shape[0]):
            df_inv.iloc[start[i]:end[i]] = df_paa.iloc[i]

        remainder = ts_size % self.window_size
        if remainder <= self.window_size / 2:
            df_inv.iloc[-remainder:] = df_paa.iloc[-1]

        return df_inv
