import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm

from utils.utils import interpolate_segments


NUM_ALPHABET_LETTERS = 26


# TODO: inherit from something like BaseApproximator
class SAX:
    """
    Symbolic Aggregate Approximation (SAX).

    :param alphabet_size: int (default = 3)
        The number of letters in the alphabet that shall be used for
        discretization. The alphabet starts from 'a' and ends with 'z' at the
        latest.
    :raises:
        ValueError: If the size of the alphabet is above 26 or below 1.
    """

    def __init__(self, alphabet_size=3):
        if NUM_ALPHABET_LETTERS < alphabet_size < 1:
            raise ValueError("The size of the alphabet needs to be between"
                             "1 (inclusive) and 26 (inclusive)")
        # TODO: super parent class
        self.alphabet_size = alphabet_size
        letters = [chr(letter) for letter
                   in range(ord('a'), ord('a') + self.alphabet_size)]
        self.alphabet = np.array(letters)

        quantiles = np.linspace(0, 1, self.alphabet_size+1)[1:-1]
        # z-value of quantiles of standard normal distribution
        self.breakpoints = norm.ppf(quantiles)

    def transform(self, df_paa):
        """
        Transform the PAA representation of each time series into its SAX
        representation (i.e. assign each PAA representation its respective
        SAX word).
        It does not modify the given PAA representations before computation,
        such as normalization. Therefore, the modification of the PAA
        representations (e.g. normalization) is the responsibility of the user.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of a time series dataset.
        :return:
            dataframe of shape (num_segments, num_ts)
        """

        # index i satisfies: breakpoints[i-1] <= paa_value < breakpoints[i]
        alphabet_idx = np.searchsorted(self.breakpoints, df_paa, side="right")
        df_sax = pd.DataFrame(data=self.alphabet[alphabet_idx],
                              index=df_paa.index, columns=df_paa.columns)
        return df_sax

    def inv_transform(self, df_sax, ts_size, window_size, symbol_mapping):
        """
        Approximate the original time series dataset by transforming its
        SAX representations into a time series dataset with the same size
        by assigning each point the symbol value of its segment.

        :param df_sax: dataframe of shape (num_segments, num_ts)
            The SAX representation of the time series dataset.
        :param ts_size: int
            The size of the original time series.
        :param window_size: int
            The size of the segments with which the given SAX representations
            were created.
        :param symbol_mapping: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the SAX symbols.
        :return:
            dataframe of shape (ts_size, num_ts)
        """

        df_mapped = symbol_mapping.get_mapped(df_sax, self.alphabet, self.breakpoints)
        df_inv = interpolate_segments(df_mapped, ts_size, window_size)
        return df_inv


class SymbolMapping(ABC):
    """
    The abstract base class for the strategy of how to map SAX symbols to
    symbol values. The class hierarchy this abstract base class creates is
    supposed to implement the 'strategy' design pattern. In that way, the
    mapping strategies can be changed and extended easily.
    """

    @abstractmethod
    def get_mapped(self, df_sax, alphabet, breakpoints):
        """
        Map the given SAX symbols to symbol values.

        :param df_sax: dataframe of shape (num_segments, num_ts)
            The SAX representation of the time series dataset.
        :param alphabet: array
            The alphabet that was used for the given SAX representations.
        :param breakpoints: array
            The breakpoints that were used for the given SAX representations.
        :return:
            dataframe of shape (num_segments, num_ts)
        """

        pass


class IntervalNormMedian(SymbolMapping):
    """
    For this mapping strategy, the median of the respective breakpoint interval
    for a standard normal distribution is used as a symbol value for the
    corresponding SAX symbol of the breakpoint interval.

    :param alphabet_size: int
        The number of letters of the alphabet that was used for creating the
        SAX representations whose symbols shall be mapped by this mapping
        strategy.
    """

    def __init__(self, alphabet_size):
        super().__init__()
        # breakpoint intervals together with their respective medians make up
        # 2 * alphabet_size intervals
        # then, the odd bounds are the quantiles for the medians within the
        # respective breakpoint interval
        median_quantiles = [bound / (2 * alphabet_size)
                            for bound in range(1, 2 * alphabet_size, 2)]
        # compute z-value of median quantiles within the respective breakpoint
        # interval for standard normal distribution in ascending order
        self.interval_medians = norm.ppf(median_quantiles)

    def get_mapped(self, df_sax, alphabet, breakpoints=None):
        mapping = dict(zip(alphabet, self.interval_medians))
        df_mapped = df_sax.replace(to_replace=mapping)
        return df_mapped


class IntervalMean(SymbolMapping):
    """
    For this mapping strategy, the mean of the respective breakpoint interval
    is used as a symbol value for the corresponding SAX symbol of the
    breakpoint interval.

    :param bound: int
        The lower and upper bound that shall be used for the lowest and
        uppermost breakpoint interval. This bound is symmetric.
        A value for these bounds is needed to avoid infinity as bounds.
    """

    def __init__(self, bound):
        super().__init__()
        self.bound = -bound if bound < 0 else bound

    def get_mapped(self, df_sax, alphabet, breakpoints):
        lower_bounds = np.insert(breakpoints, 0, -self.bound)
        upper_bounds = np.append(breakpoints, self.bound)
        interval_means = [(lower_bound + upper_bound) / 2
                          for lower_bound, upper_bound
                          in zip(lower_bounds, upper_bounds)]
        mapping = dict(zip(alphabet, interval_means))
        df_mapped = df_sax.replace(to_replace=mapping)
        return df_mapped


class ValuePoints(SymbolMapping):
    """
    The abstract class for the strategies of mapping the symbol value to a SAX
    symbol by deriving metrics from all original normalized time series points
    based on their respective breakpoint interval they are located in.
    This abstract class is supposed to implement the 'template method' design
    pattern. In that way, such strategies can be changed and extended easily.

    Caveat: It could be the case that there is a symbol in a SAX representation
    for that there is no symbol value with such mapping strategy, because there
    is no time series point in the respective interval, at all.
    But, it is very unlikely that this will be the case, because of the
    equiprobable breakpoint intervals of the standard normal distribution;
    especially not for realistic real-life and long time series.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset the SAX representations whose
        symbols shall be mapped by such a mapping strategy belong to.
    """

    def __init__(self, df_norm):
        super().__init__()
        self.df_norm = df_norm

    def get_mapped(self, df_sax, alphabet, breakpoints):
        df_mapped = pd.DataFrame(index=df_sax.index, columns=df_sax.columns)
        idx = 0
        # iteratively for each time series, because different means in
        # breakpoint intervals for each
        for col_name, col_data in self.df_norm.items():
            # assign respective alphabet index to each point in time series
            alphabet_idx = np.searchsorted(breakpoints,
                                           col_data, side="right")
            symbol_values = self.get_symbol_values(col_data.groupby(by=alphabet_idx))

            mapping = {}
            for alphabet_idx, symbol_value in symbol_values.items():
                mapping.update({alphabet[alphabet_idx]: symbol_value})
            df_mapped.iloc[:, idx] = df_sax.iloc[:, idx].replace(to_replace=mapping)
            idx += 1

        return df_mapped

    @abstractmethod
    def get_symbol_values(self, grouped_by_symbol):
        """
        The hook method that shall be overridden by subclasses.
        Determine what metric shall be used and compute the corresponding
        symbol values for the SAX symbols based on that metric.

        :param grouped_by_symbol: pd.SeriesGroupBy
            The groupby object that contains information about the grouping of
            the original normalized time series points based on their
            respective SAX symbol.
        :return:
            pd.Series of shape (num_SAX_symbols,)
        """

        pass


class MeanValuePoints(ValuePoints):
    """
    For this mapping strategy, the symbol value of a SAX symbol is computed by
    the mean value of all original normalized time series points that are
    located in the respective breakpoint interval of the SAX symbol.
    This mapping strategy is equivalent to fitting a normal distribution for
    each breakpoint interval and taking the respective mean as symbol value.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset the SAX representations whose
        symbols shall be mapped by this mapping strategy belong to.
    """

    def __init__(self, df_norm):
        super().__init__(df_norm=df_norm)

    def get_symbol_values(self, grouped_by_symbol):
        return grouped_by_symbol.mean()


class MedianValuePoints(ValuePoints):
    """
    For this mapping strategy, the symbol value of a SAX symbol is computed by
    the median value of all original normalized time series points that are
    located in the respective breakpoint interval of the SAX symbol.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset the SAX representations whose
        symbols shall be mapped by this mapping strategy belong to.
    """

    def __init__(self, df_norm):
        super().__init__(df_norm=df_norm)

    def get_symbol_values(self, grouped_by_symbol):
        return grouped_by_symbol.median()
