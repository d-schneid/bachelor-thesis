import numpy as np
import pandas as pd

from utils.utils import constant_segmentation, interpolate_segments
from discretization.sax.sax import SAX
from discretization.sax.abstract_sax import (AbstractSAX,
                                             NUM_ALPHABET_LETTERS, breakpoints)


NUMERATOR_VAR_SLOPE = 0.03


def _compute_slopes(df_norm, df_paa, window_size):
    ts_size = df_norm.shape[0]
    start, end, num_segments = constant_segmentation(ts_size, window_size)
    time_means = pd.DataFrame(data=[np.mean(range(start, end))
                                    for start, end in zip(start, end)])
    time_means = interpolate_segments(time_means, ts_size, window_size).squeeze()
    time_diff_points = pd.Series(df_norm.index.values) - time_means

    # contains the sum for each segment that is used in the numerator to
    # compute the slope of each segment
    sums_numerator = []
    # contains the sum for each segment that is used in the denumerator to
    # compute the slope of each segment
    sums_denumerator = []
    for i_segment in range(num_segments):
        time_diff_segment = time_diff_points[start[i_segment]:end[i_segment]]

        point_segment = df_norm.iloc[start[i_segment]:end[i_segment]]
        time_diff_point_segment = point_segment.mul(time_diff_segment, axis=0)
        segment_sum_numerator = np.sum(time_diff_point_segment, axis=0)
        sums_numerator.append(segment_sum_numerator)

        segment_sum_denumerator = np.sum(time_diff_segment.pow(2))
        sums_denumerator.append(segment_sum_denumerator)

    slopes_numerator = pd.DataFrame(data=np.array(sums_numerator),
                                    index=df_paa.index,
                                    columns=df_paa.columns)
    slopes_denumerator = pd.Series(data=sums_denumerator)
    slopes = slopes_numerator.divide(slopes_denumerator, axis=0)
    return slopes


class OneDSAX(AbstractSAX):

    def __init__(self, alphabet_size_avg=3, alphabet_size_slope=3, var_slope=None):
        if NUM_ALPHABET_LETTERS < alphabet_size_slope < 1:
            raise ValueError(f"The size of an alphabet needs to be between "
                             f"1 (inclusive) and {NUM_ALPHABET_LETTERS} (inclusive)")
        super().__init__(alphabet_size_avg=alphabet_size_avg)

        self.alphabet_size_slope = alphabet_size_slope
        letters_slope = [chr(letter) for letter
                         in range(ord('a'), ord('a') + self.alphabet_size_slope)]
        self.alphabet_slope = np.array(letters_slope)
        # for slope different variance of Gaussian distribution is possible
        self.var_slope = var_slope
        # for breakpoints of slope, window size need to be known
        # breakpoints will be determined later when window size is known

    def _fit(self, window_size):
        # need to be here, because at construction, window size not known
        if self.var_slope is None:
            self.var_slope = np.sqrt(NUMERATOR_VAR_SLOPE / window_size)
        self.breakpoints_slope = breakpoints(self.alphabet_size_slope, scale=self.var_slope)

    def transform(self, df_norm, df_paa, window_size):
        df_avg = SAX(self.alphabet_size_avg).transform(df_paa)

        # slope for each segment for each time series
        slopes = _compute_slopes(df_norm, df_paa, window_size)
        self._fit(window_size)
        alphabet_slope_idx = np.searchsorted(self.breakpoints_slope, slopes, side="right")
        df_slope = pd.DataFrame(data=self.alphabet_slope[alphabet_slope_idx],
                                index=slopes.index, columns=slopes.columns)

        return df_avg + df_slope
