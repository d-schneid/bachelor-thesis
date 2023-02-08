import math
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm


# the number of symbols in the Latin alphabet
NUM_ALPHABET_SYMBOLS = 26
# assumed number of bits that are needed to store one time series point as
# a floating-point number
BITS_PER_TS_POINT = 64


def breakpoints(alphabet_size, scale=1):
    """
    Compute equiprobable breakpoints under a Gaussian distribution for
    quantizing raw values.

    :param alphabet_size: int
        The number of equiprobable regions under the Gaussian distribution.
    :param scale: float
        The variance of the Gaussian distribution.
    :return:
        np.array of shape (alphabet_size-1,)
    """

    quantiles = [numerator / alphabet_size for numerator in range(1, alphabet_size)]
    # z-values of quantiles of Gaussian distribution with variance 'scale'
    breakpts = norm.ppf(quantiles, scale=scale)
    return breakpts


def linearize_sax_word(df_sax, symbols_per_segment):
    """
    Linearize SAX representations that consist of multiple symbols per segment.

    :param df_sax: dataframe of shape (num_segments, num_ts)
        The SAX representations that shall be linearized.
    :param symbols_per_segment: int
        The symbols per segment that are used in the given SAX representations.
    :return:
        dataframe of shape (num_segments * symbols_per_segment, num_ts)
    """

    symbols_splits = []
    for i in range(symbols_per_segment):
        symbols_splits.append(df_sax.applymap(lambda symbols: symbols[i]))

    # use mergesort (stable) to preserve order of symbols between dataframes
    return pd.concat(symbols_splits).sort_index(kind="mergesort")


class AbstractSAX(ABC):
    """
    The abstract class from that all SAX variants (SAX, 1d-SAX, aSAX, eSAX)
    inherit.

    :param alphabet_size: int (default = 3)
        The number of symbols in the alphabet that shall be used for
        discretization. The alphabet starts from 'a' and ends with 'z' at the
        latest.
    :raises:
        ValueError: If the size of the alphabet is above 26 or below 1.
    """

    def __init__(self, alphabet_size=3):
        if alphabet_size > NUM_ALPHABET_SYMBOLS or alphabet_size < 1:
            raise ValueError(f"The size of an alphabet needs to be between "
                             f"1 (inclusive) and {NUM_ALPHABET_SYMBOLS} (inclusive)")
        self.alphabet_size = alphabet_size
        self.bits_per_symbol = math.ceil(np.log2(self.alphabet_size))
        symbols = [chr(symbol) for symbol
                   in range(ord('a'), ord('a') + self.alphabet_size)]
        self.alphabet = np.array(symbols)
        self.breakpoints = breakpoints(self.alphabet_size)
        self.symbols_per_segment = 1

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Transform the PAA representation of each time series into its symbolic
        representation corresponding to the respective SAX variant (i.e. assign
        each PAA representation its respective symbolic word).
        It does not modify the given PAA representations before computation,
        such as normalization. Therefore, the modification of the PAA
        representations (e.g. normalization) is the responsibility of the user.

        :param args:
            Parameters needed for the transformation based on the respective
            SAX variant, such as PAA representations of the original time
            series dataset.
        :param kwargs:
            Parameters needed for the transformation based on the respective
            SAX variant, such as PAA representations of the original time
            series dataset.
        :return:
            The symbolic representation of the PAA representation corresponding
            to the respective SAX variant.
        """

        pass

    @abstractmethod
    def inv_transform(self, *args, **kwargs):
        """
        Approximate the original time series dataset by transforming its
        symbolic representations corresponding to a SAX variant into a time
        series dataset with the same size by assigning each point the value of
        its corresponding symbol.

        :param args:
            Parameters needed for the inverse transformation based on the
            respective SAX variant, such as symbolic representations of the
            original time series dataset based on the corresponding SAX
            variant.
        :param kwargs:
            Parameters needed for the inverse transformation based on the
            respective SAX variant, such as symbolic representations of the
            original time series dataset based on the corresponding SAX
            variant.
        :return:
            A time series dataset with the same size as the original time
            series dataset and computed based on its symbolic representations
            corresponding to the respective SAX variant.
        """

        pass

    @abstractmethod
    def transform_inv_transform(self, df_paa, df_norm, window_size, df_breakpoints=None, **symbol_mapping):
        """
        Transform the PAA representation of each time series into its symbolic
        representation and inverse transform these symbolic representations.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of a time series dataset that shall be
            transformed into their symbolic representations.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series dataset that was used to create the given PAA
            representations
        :param window_size: int
            The size of the window that was used to create the given PAA
            representations.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            Can only be used for the aSAX. Ignored for other SAX variants.
            The individual breakpoints for the given PAA representations that
            shall be used to transform them into their symbolic
            representations.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            This parameter is intended to allow breakpoints based on the
            k-means clustering of the original normalized time series data
            points.
        :param symbol_mapping: multiple objects of SymbolMapping
            The symbol mapping strategies that shall be used to inverse
            transform the symbolic representations of the given PAA
            representations.
            The appropriate symbol mapping strategies depend on the used SAX
            variant.
        :return:
            dataframe of shape (ts_size, num_ts)
        """

        pass

    @abstractmethod
    def transform_to_symbolic_ts(self, df_paa, df_norm, window_size, df_breakpoints=None):
        """
        Transform the given PAA representations of time series into their
        symbolic representation and interpolate this symbolic representation
        to the whole time series.
        This method is especially used for the computation of the inconsistency
        metrics of the different SAX variants.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of the time series that shall be
            transformed.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series that belong to the given PAA representations.
        :param window_size: int
            The size of the window that was used to create the given PAA
            representations.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            Can only be used for the aSAX and is ignored for other SAX
            variants.
            The individual breakpoints for the PAA representations of the given
            time series dataset that shall be used to transform the respective
            PAA representation into its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            This parameter is intended to allow breakpoints based on the
            k-means clustering of the original normalized time series data
            points.
        :return: dataframe of shape (ts_size, num_ts)
        """

        pass

    def adjust_symbolic_ts(self, df_symbolic_ts):
        """
        Adjust the given symbolic time series.
        This is only needed for the 1d-SAX and does not have any effect for the
        other SAX variants.

        :param df_symbolic_ts: dataframe of shape (ts_size, num_ts)
            The symbolic time series that shall be adjusted.
        :return: dataframe of shape (ts_size, num_ts)
        """

        return df_symbolic_ts

    def compute_compression_ratio_percentage(self, ts_size, num_segments):
        """
        Compute the compression ratio in percentage. The compression ratio is
        the ratio of the number of bits needed to store the symbolic
        representation of a time series and the number of bits needed to store
        the raw values of the corresponding time series.

        :param ts_size: int
            The size of the time series for that the compression ratio shall be
            calculated.
        :param num_segments: int
            The number of (symbolic) points that the time series for that the
            compression ratio shall be calculated shall have after transforming
            it into its symbolic representation.
        :return: float
        """

        num_bits_ts = BITS_PER_TS_POINT * ts_size
        num_symbols = self.symbols_per_segment * num_segments
        num_bits_symbolic = self.bits_per_symbol * num_symbols
        return (num_bits_symbolic / num_bits_ts) * 100

    def _transform_to_symbolic_repr_only(self, df_paa, df_norm, window_size, df_breakpoints):
        """
        Wrapper for the transformation of the time series based on the
        respective SAX variant. Only returns the symbolic representations of
        the time series that result from the transformation and nothing else.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of the given normalized time series
            dataset.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The normalized time series dataset that shall be transformed into
            its symbolic representations.
        :param window_size: int
            The size of the window that was used to create the given PAA
            representations.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
            Is ignored for all SAX variants except the aSAX.
            The individual breakpoints for the given PAA representations of the
            given time series dataset that shall be used to transform the
            respective PAA representation into its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            With this parameter breakpoints based on the k-means clustering of
            the original normalized time series data points are also possible.
        :return: dataframe of shape (num_segments, num_ts)
        """

        return self.transform(df_paa=df_paa, df_norm=df_norm, window_size=window_size,
                              df_breakpoints=df_breakpoints)

    def transform_to_symbolic_repr_histogram(self, df_paa, df_norm, window_size, df_breakpoints):
        """
        Hook method for the transformation of the time series into the symbolic
        representation that is used to compute histograms for the computation
        of the Kullback-Leibler divergence based on the respective SAX variant.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of the given normalized time series
            dataset.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The normalized time series dataset that shall be transformed into
            its symbolic representations.
        :param window_size: int
            The size of the window that was used to create the given PAA
            representations.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
            Is ignored for all SAX variants except the aSAX.
            The individual breakpoints for the given PAA representations of the
            given time series dataset that shall be used to transform the
            respective PAA representation into its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            With this parameter breakpoints based on the k-means clustering of
            the original normalized time series data points are also possible.
        :return: dataframe
            The symbolic representations of the given time series dataset based
            on the respective SAX variant that are used to compute histograms
            for the computation of the Kullback-Leibler divergence.
        """

        return self._transform_to_symbolic_repr_only(df_paa, df_norm, window_size, df_breakpoints)

    def get_histogram_bins(self):
        """
        This is a hook method that determines the alphabet that shall be used
        for building histograms for the computation of the Kullback-Leibler
        divergence between raw time series and their symbolic representations.

        :return: np.array
            The alphabet of the respective SAX variant except for the 1d-SAX.
            For the 1d-SAX, it returns the cartesian product of the alphabet
            for the segment means and the alphabet for the segement slopes.
        """

        return self.alphabet

    def compute_raw_bin_idxs(self, df_norm, hist_binning):
        """
        This is a hook method that assigns each point for each time series to
        the histogram bin whose breakpoints correspond to the value of the
        respective point.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset for that the histogram
            shall be built.
        :param hist_binning: HistogramBinning
            The strategy with which the bins for the histogram shall be built.
        :return: dataframe of shape (ts_size, num_ts)
            Each point of each given time series is assigned its corresponding
            bin id in the form of a symbol of the respective alphabet based on
            the SAX variant.
        """

        return hist_binning.assign_histogram_bins(df_norm, self.alphabet)
