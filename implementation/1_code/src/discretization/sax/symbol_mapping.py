import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm


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
    :param var: float (default = 1)
        The variance of the Gaussian distribution used to compute the
        breakpoints for the intervals for the quantization of raw values.
        Usually, this value only deviates from the set default value for the
        quantization of the slope values in the 1d-SAX discretization method.
    """

    def __init__(self, alphabet_size, var=1):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.var = var
        # breakpoint intervals together with their respective medians make up
        # 2 * alphabet_size intervals
        # then, the odd bounds are the quantiles for the medians within the
        # respective breakpoint interval
        median_quantiles = [bound / (2 * self.alphabet_size)
                            for bound in range(1, 2 * self.alphabet_size, 2)]
        # compute z-value of median quantiles within the respective breakpoint
        # interval for standard normal distribution in ascending order
        self.interval_medians = norm.ppf(median_quantiles, scale=self.var)

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
        mapped = []
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
            mapped.append(df_sax.iloc[:, idx].replace(to_replace=mapping))
            idx += 1

        return pd.concat(mapped, axis=1)

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
