import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import entropy


# to avoid relative frequencies of zero in the discrete probability function to
# be able to compute the Kullback-Leibler divergence
SMOOTHING_PARAM = 1e-10


class HistogramBinning(ABC):
    """
    This is the abstract class for a histogram binning strategy. A histogram
    binning strategy determines the way the bins for a histogram are built.
    """

    @abstractmethod
    def assign_histogram_bins(self, df_ts, alphabet):
        """
        Assign each point of each given time series its corresponding bin of
        the histogram that shall be built based on the chosen histogram binning
        strategy.

        :param df_ts: dataframe of shape (ts_size, num_ts)
            The time series whose points shall be binned for building a
            histogram.
        :param alphabet: np.array of shape (num_symbols,)
            The alphabet that shall be used for building the bins of a
            histogram. The histogram will have 'num_symbols' bins and the
            symbols of the alphabet represent the respective bin id in
            ascending order.
        :return: dataframe of shape (ts_size, num_ts)
            Each point of each given time series is assigned its corresponding
            bin id in the form of a symbol of the given alphabet.
        """

        pass


class EqualWidthBinning(HistogramBinning):
    """
    The bins of the histogram that shall be built with this strategy all have
    the same width based on the number of bins (i.e. the size of the given
    alphabet). The first bin starts at the minimum point and the last bin ends
    at the maximum point of the respective time series.
    With this strategy, the distribution of the given time series is estimated
    by the built histogram.
    """

    def assign_histogram_bins(self, df_ts, alphabet):
        assigned_bin_idxs = []
        for i in range(df_ts.shape[1]):
            current_ts = df_ts.iloc[:, i]
            bin_breakpoints = np.linspace(current_ts.min(), current_ts.max(),
                                          alphabet.size + 1)[1:-1]
            bin_idxs = np.searchsorted(bin_breakpoints, current_ts, side="right")
            assigned_bin_idxs.append(pd.Series(bin_idxs))

        df_bin_idxs = pd.concat(assigned_bin_idxs, axis=1)
        mapping = {bin_idx: bin_symbol for bin_idx, bin_symbol in enumerate(alphabet)}

        return df_bin_idxs.replace(mapping)


class QuantileBinning(HistogramBinning):
    """
    All bins of the histogram that shall be built with this strategy
    approximately contain the same number of points of the respective time
    series. The breakpoints of the bins are the quantiles of the respective
    time series based on the size of the given alphabet.
    """

    def assign_histogram_bins(self, df_ts, alphabet):
        quantiles = df_ts.quantile(np.linspace(0, 1, alphabet.size + 1)).iloc[1:-1, :]

        assigned_bin_idxs = []
        for i in range(df_ts.shape[1]):
            bin_idxs = np.searchsorted(quantiles.iloc[:, i], df_ts.iloc[:, i], side="right")
            assigned_bin_idxs.append(pd.Series(bin_idxs))

        df_bin_idxs = pd.concat(assigned_bin_idxs, axis=1)
        mapping = {bin_idx: bin_symbol for bin_idx, bin_symbol in enumerate(alphabet)}

        return df_bin_idxs.replace(mapping)


def _compute_histogram(df_bin_idxs, sax_variant):
    """
    Compute the histogram based on the points that are assigned to their
    respective histogram bin. Then, the relative frequencies of the histogram
    bins are computed resulting in a discrete probability distribution.

    :param df_bin_idxs: dataframe of shape (ts_size, num_ts)
        For each time series each point is assigned to its histogram bin it
        belongs to in the histogram that shall be built.
    :param sax_variant: AbstractSAX
        The SAX variant that is used to build the histograms for the
        corresponding symbolic representations.
    :return: dataframe of shape (alphabet_size, num_ts)
        The histogram for each time series with the relative frequency for each
        bin. The relative frequencies of a histogram sum up to 1.
    """

    # sorted in lexicographic order
    df_hist_abs = df_bin_idxs.apply(lambda ts: ts.value_counts().sort_index())
    # all possible bins based on the given 'sax_variant' shall be contained in
    # the histogram, therefore add bins where no value is located
    df_hist_abs = df_hist_abs.reindex(sax_variant.get_histogram_bins()).fillna(0)
    # smoothing parameter to avoid 0 in computation of Kullback-Leibler
    # divergence
    df_hist_abs = df_hist_abs + SMOOTHING_PARAM
    df_hist_rel = df_hist_abs / df_hist_abs.sum()

    return df_hist_rel


def _compute_raw_histogram(df_norm, sax_variant, hist_binning):
    """
    Compute the histogram of the given time series. Each time series point is
    assigned to the histogram bin whose breakpoints correspond to the value of
    the point. Then, the relative frequencies of the histogram bins are
    computed resulting in a discrete probability distribution.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The original normalized time series dataset for that the histogram
        shall be built.
    :param sax_variant: AbstractSAX
        The SAX variant that is used to build the histogram based on the
        corresponding symbolic representations of the time series.
    :param hist_binning: HistogramBinning
        The strategy with which the bins for the histogram shall be built.
    :return: dataframe of shape (alphabet_size, num_ts)
        The histogram for each time series with the relative frequency for each
        bin. The relative frequencies of a histogram sum up to 1.
    """

    df_bin_idxs = sax_variant.compute_raw_bin_idxs(df_norm, hist_binning)
    return _compute_histogram(df_bin_idxs, sax_variant)


def _compute_symbolic_histogram(df_paa, df_norm, window_size, sax_variant, df_breakpoints):
    """
    Compute the histogram of the symbolic representations of the given time
    series. Each symbol of the corresponding alphabet of the given
    'sax_variant' makes up a separate bin. Then, the relative frequency of each
    bin is computed by the number of respective symbols compared to the total
    number of symbols in the respective symbolic representation. This results
    in a discrete probability distribution for the symbols contained in the
    respective symbolic representation.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations of the given normalized time series
        dataset.
    :param df_norm: dataframe of shape (ts_size, num_ts)
        The original normalized time series dataset for that the histograms
        shall be built for its symbolic representations.
    :param window_size: int
        The size of the window that was used to create the given PAA
        representations.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
        Is ignored for all SAX variants except the aSAX.
        The individual breakpoints for the PAA representations of the given
        time series dataset that shall be used to transform the respective
        PAA representation into its aSAX representation.
        If None, the respective breakpoints resulting from the k-means
        clustering of the respective PAA points are used.
        With this parameter breakpoints based on the k-means clustering of
        the original normalized time series data points are also possible.
    :return: dataframe of shape (alphabet_size, num_ts)
        The histogram of the symbolic representation for each time series with
        the relative frequency for each bin. The relative frequencies of a
        histogram sum up to 1.
    """

    # symbols are used for bin id
    df_bin_idxs = sax_variant.transform_to_symbolic_repr_histogram(df_paa, df_norm,
                                                                   window_size, df_breakpoints)
    return _compute_histogram(df_bin_idxs, sax_variant)


def compute_kullback_leibler_divergence(df_paa, df_norm, window_size, sax_variant,
                                        hist_binning, df_breakpoints=None):
    """
    Compute the Kullback-Leibler divergence based on the binary logarithm
    between the discrete probability functions of a time series and its
    symbolic representation based on a chosen SAX variant.
    The discrete probability functions are computed based
    on histograms where the number of bins is equal to the respective alphabet
    size of the chosen SAX variant.
    For the time series, each point is assigned to the histogram bin where
    its value is located between the bin breakpoints. These bin breakpoints are
    computed based on the chosen histogram binning strategy.
    For the symbolic representation of the time series, the histogram bins
    consist of the (relative) number of the alphabet symbols.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations of the given time series dataset.
    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series for which the Kullback-Leibler divergence between
        themselves and their symbolic representations shall be computed.
    :param window_size: int
        The size of the window that was used to create the given PAA
        representations.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to compute the symbolic
        representations of the given time series and with that the histograms
        shall be computed.
    :param hist_binning: HistogramBinning
        The strategy with which the bins for the histograms of the given time
        series shall be built.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
        Is ignored for all SAX variants except the aSAX.
        The individual breakpoints for the PAA representations of the given
        time series dataset that shall be used to transform the respective
        PAA representation into its aSAX representation.
        If None, the respective breakpoints resulting from the k-means
        clustering of the respective PAA points are used.
        With this parameter breakpoints based on the k-means clustering of
        the original normalized time series data points are also possible.
    :return: pd.Series of shape (num_ts,)
        The score of the Kullback-Leibler divergence for each time series and
        its symbolic representation.
    """

    df_symbolic_hist = _compute_symbolic_histogram(df_paa, df_norm, window_size,
                                                   sax_variant, df_breakpoints)
    df_raw_hist = _compute_raw_histogram(df_norm, sax_variant, hist_binning)

    return pd.Series(entropy(df_raw_hist, df_symbolic_hist, base=2))
