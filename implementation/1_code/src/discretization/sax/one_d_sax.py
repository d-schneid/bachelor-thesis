import math
import warnings
import itertools
import numpy as np
import pandas as pd

from utils import constant_segmentation, interpolate_segments, scale_min_max
from discretization.symbol_mapping import ValuePoints, IntervalNormMedian
from discretization.sax.sax import SAX
from discretization.sax.abstract_sax import AbstractSAX, breakpoints
from discretization.abstract_discretization import (
    NUM_ALPHABET_SYMBOLS, BITS_PER_TS_POINT, _get_alphabet)


# the value of the numerator of the variance function given in [1] for the
# Gaussian distribution that is used for quantizing the slope values
NUMERATOR_VAR_SLOPE = 0.03


def compute_slopes(df_norm, window_size):
    """
    Compute the slope of the linear function fit by linear regression for each
    segment in each time series.
    The computation is based on the formula for the slope value given in [1].

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset that shall be transformed into its
        1d-SAX representations.
    :param window_size: int
        The size of the segments with which the given PAA representations were
        created.
    :return:
        dataframe of shape (num_segments, num_ts)
    """

    if window_size == 1:
        raise ValueError("'window_size' must be greater than 1, because the "
                         "slope of a single point is not defined.")

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
                                    index=range(num_segments),
                                    columns=df_norm.columns)

    slopes_denumerator = pd.Series(data=sums_denumerator)
    slopes = slopes_numerator.divide(slopes_denumerator, axis=0)
    return slopes


class OneDSAX(AbstractSAX):
    """
    One-D Symbolic Aggregate Approximation (1d-SAX).

    :param alphabet_size_avg: int (default = 3)
        The number of symbols in the alphabet that shall be used for
        discretizing the average values of the segments. The alphabet starts
        from 'a' and ends with 'z' at the latest.
    :param alphabet_size_slope: int (default = 3)
        The number of symbols in the alphabet that shall be used for
        discretizing the slope values of the segments. The alphabet starts
        from 'a' and ends with 'z' at the latest.
    :param var_slope: float or None (default = None)
        The variance of the Gaussian distribution used to quantize the slope
        values of the segments.
        If None, the square root of 0.03/'window_size' is used as proposed in [1].
    :raises:
        ValueError: If the size of one of the alphabets is above 26 or below 1.

    References
    ----------
    [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a Novel
    Symbolic Representation for Time Series. IDA 2013.
    """

    def __init__(self, alphabet_size_avg=3, alphabet_size_slope=3, var_slope=None):
        #if alphabet_size_slope > NUM_ALPHABET_SYMBOLS or alphabet_size_slope < 1:
         #   raise ValueError(f"The size of an alphabet needs to be between "
          #                   f"1 (inclusive) and {NUM_ALPHABET_SYMBOLS} (inclusive)")
        super().__init__(alphabet_size=alphabet_size_avg)
        self.name = "1d-SAX"

        self.alphabet_size_avg = self.alphabet_size
        self.bits_per_symbol_avg = self.bits_per_symbol
        self.alphabet_avg = self.alphabet
        self.breakpoints_avg = self.breakpoints

        self.alphabet_size_slope = alphabet_size_slope
        self.bits_per_symbol_slope = math.ceil(np.log2(self.alphabet_size_slope))
        self.alphabet_slope = _get_alphabet(self.alphabet_size_slope)
        self.var_slope = var_slope
        self.symbols_per_segment = 2
        # breakpoints for slope values of the segments are determined during
        # 1d-SAX transformation, when 'window_size' is known
        # can also be set with the method 'set_breakpoints_slope'
        self.breakpoints_slope = None

    def set_breakpoints_slope(self, window_size):
        """
        Set the breakpoints for quantizing the slope values of the segments
        based on a Gaussian distribution whose variance is a decreasing
        function of the 'window_size'.

        :param window_size: int
            The size of the segments with which the preprocessed PAA
            representations were created.
        :return: None
        """

        if self.var_slope is None:
            self.var_slope = np.sqrt(NUMERATOR_VAR_SLOPE / window_size)
        self.breakpoints_slope = breakpoints(self.alphabet_size_slope,
                                             scale=self.var_slope)

    def transform(self, df_paa, df_norm, window_size, *args, **kwargs):
        """
        Transform the normalized time series dataset into its 1d-SAX
        representations (i.e. assign each time series its respective 1d-SAX
        word).

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of the given normalized time series dataset.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The normalized time series dataset that shall be transformed into
            its 1d-SAX representations.
        :param window_size: int
            The size of the segments with which the given PAA representations
            were created.
        :return: dataframe of shape (num_segments, num_ts)
            The returned dataframe contains the 1d-SAX representation of each
            time series. Thereby, each segment of a time series is assigned two
            symbols. The first symbol is the quantized mean and the second is
            the quantized slope of the respective segment.
        """

        df_avg = SAX(self.alphabet_size_avg).transform(df_paa)

        slope_values = compute_slopes(df_norm, window_size)
        self.set_breakpoints_slope(window_size)
        # index i satisfies: breakpoints_slope[i-1] <= slope_value < breakpoints_slope[i]
        alphabet_slope_idx = np.searchsorted(self.breakpoints_slope, slope_values, side="right")
        df_slope = pd.DataFrame(data=self.alphabet_slope[alphabet_slope_idx],
                                index=slope_values.index,
                                columns=slope_values.columns)

        # String concatenation
        df_one_d_sax = df_avg + df_slope
        return df_one_d_sax

    def inv_transform(self, df_norm, df_one_d_sax, window_size,
                      symbol_mapping_avg, symbol_mapping_slope, *args, **kwargs):
        """
        Approximate the original time series dataset by transforming its 1d-SAX
        representations into a time series dataset with the same size. Each
        point is assigned a value combined from the symbol value of its
        segment average and the symbol value of its segment slope with the
        formula:
        symbol_value_avg + symbol_value_slope * (time_point - time_segment_middle).

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The normalized time series dataset that has been transformed into
            its 1d-SAX representations.
        :param df_one_d_sax: dataframe of shape (num_segments, num_ts)
            The 1d-SAX representations of the given normalized time series
            dataset.
        :param window_size: int
            The size of the segments with which the PAA representations of the
            given normalized time series dataset for its 1d-SAX transformation
            were created.
        :param symbol_mapping_avg: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the 1d-SAX symbols into which the segment averages were quantized.
        :param symbol_mapping_slope: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the 1d-SAX symbols into which the segment slopes were quantized.
            Caveat: In case of one of the 'ValuePoints' strategies, the results
            might only be meaningful if the segment slopes ('compute_slopes'
            function) were used for its initialization.
        :return:
            dataframe of shape (ts_size, num_ts)
        """

        if isinstance(symbol_mapping_slope, ValuePoints):
            warnings.warn("Make sure you used the segment slopes for "
                          "initializing the chosen 'symbol_mapping_slope' "
                          "strategy. Otherwise, the results might not be "
                          "meaningful.")
        if isinstance(symbol_mapping_slope, IntervalNormMedian):
            warnings.warn("Make sure you have initialized the chosen "
                          "'symbol_mapping_slope' strategy with the variance "
                          "of the Gaussian distribution that was used to "
                          "determine the breakpoint intervals for the segment "
                          "slopes.")

        ts_size = df_norm.shape[0]
        start, end, num_segments = constant_segmentation(ts_size, window_size)
        time_segment_middle = pd.DataFrame(data=(start + end - 1) / 2)
        time_segment_middle = interpolate_segments(time_segment_middle, ts_size,
                                                   window_size).squeeze()

        df_avg = df_one_d_sax.applymap(lambda symbols: symbols[0])
        df_slope = df_one_d_sax.applymap(lambda symbols: symbols[1])

        df_mapped_avg = symbol_mapping_avg.get_mapped(df_avg, self.alphabet_avg,
                                                      self.breakpoints_avg)
        df_avg = interpolate_segments(df_mapped_avg, ts_size, window_size)

        df_mapped_slope = symbol_mapping_slope.get_mapped(df_slope,
                                                          self.alphabet_slope,
                                                          self.breakpoints_slope)
        # when symbol mapping strategy 'EncodedMinMaxScaling' is used for the
        # segment means, they are transformed to the range [0, 1]
        # therefore, the segment slopes need to be adjusted accordingly
        if kwargs.get("scale_slopes", False):
            df_mapped_slope /= (df_norm.max() - df_norm.min())
        df_slope = interpolate_segments(df_mapped_slope, ts_size, window_size)

        time_diff = pd.Series(df_norm.index.values) - time_segment_middle
        df_slope_time_diff = df_slope.mul(time_diff, axis=0)

        df_inv = df_avg + df_slope_time_diff
        # due to the separate transformations of segment means and segment
        # slopes, points can be slightly out of the range of [0, 1]
        if kwargs.get("scale_slopes", False):
            return scale_min_max(df_inv, df_inv.min(), df_inv.max())
        return df_inv

    def transform_inv_transform(self, df_paa, df_norm, window_size, df_breakpoints=None, **symbol_mapping):
        df_one_d_sax = self.transform(df_paa, df_norm, window_size)
        return self.inv_transform(df_norm, df_one_d_sax, window_size, **symbol_mapping)

    def transform_to_symbolic_ts(self, df_paa, df_norm, window_size, df_breakpoints=None):
        df_one_d_sax = self.transform(df_paa, df_norm, window_size)
        return interpolate_segments(df_one_d_sax, df_norm.shape[0], window_size)

    def adjust_symbolic_ts(self, df_symbolic_ts):
        """
        Adjust the given symbolic time series due to the separate alphabet for
        slope values.
        Without any adjustment, the same time series point mapped to the same
        symbol for its respective segment means and different symbols for its
        respective segment slopes would be considered as discretized into two
        different bins.
        But, this is not true if the different symbols for the segment slopes
        are opposites of each other. Meaning one symbol describes decreasing
        characteristics while the other describes the opposite increasing
        characteristics.
        Therefore, this method accounts for this fact by transforming every
        two opposite slope symbols into one common symbol.

        :param df_symbolic_ts: dataframe of shape (ts_size, num_ts)
            The symbolic time series that shall be adjusted.
        :return: dataframe of shape (ts_size, num_ts)
        """

        if self.alphabet_size_slope == 1:
            return df_symbolic_ts
        alphabet_mid_idx = round(self.alphabet_size_slope / 2) - 1
        return df_symbolic_ts.applymap(self._adjust_slope_symbol, alphabet_mid_idx=alphabet_mid_idx)

    def _adjust_slope_symbol(self, segment_symbols, alphabet_mid_idx):
        """
        Check the slope symbol of every segment and convert to its opposite
        symbol in the alphabet for the slope values if its index is larger than
        the index of the symbol in the middle of this alphabet.
        This method is applied to every cell of the dataframe that contains the
        symbolic time series.

        :param segment_symbols: str of len = 2
            The first character is the symbol for the respective segment mean,
            the second character is the symbol for the respective segment
            slope.
        :param alphabet_mid_idx: int
            The index of the symbol in the middle of the alphabet for the slope
            values.
        :return: str of len = 2
            The symbol of the respective segment mean and the (possibly
            converted) symbol of the respective segment slope.
        """

        avg_symbol = segment_symbols[0]
        slope_symbol = segment_symbols[1]
        slope_symbol_idx = np.where(self.alphabet_slope == slope_symbol)[0][0]
        if slope_symbol_idx > alphabet_mid_idx:
            slope_symbol_idx = self.alphabet_size_slope - 1 - slope_symbol_idx
            slope_symbol = self.alphabet_slope[slope_symbol_idx]
        return avg_symbol + slope_symbol

    def compute_compression_ratio_percentage(self, ts_size, num_segments):
        num_bits_ts = BITS_PER_TS_POINT * ts_size
        num_bits_symbolic = (self.bits_per_symbol_avg + self.bits_per_symbol_slope) * num_segments
        return (num_bits_symbolic / num_bits_ts) * 100

    def get_histogram_bins(self):
        return np.array(["".join(symbols_avg_slope) for symbols_avg_slope in
                         itertools.product(self.alphabet_avg, self.alphabet_slope)])

    def compute_raw_bin_idxs(self, df_norm, hist_binning):
        df_bin_idxs_avg = hist_binning.assign_histogram_bins(df_norm.tail(-1), self.alphabet_avg)
        # use first differences as 'true' slopes of time series points
        df_first_diffs = (df_norm - df_norm.shift(1)).tail(-1)
        df_bin_idxs_slope = hist_binning.assign_histogram_bins(df_first_diffs, self.alphabet_slope)
        df_bin_idxs = df_bin_idxs_avg + df_bin_idxs_slope

        return df_bin_idxs
