import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm

from discretization.sax.abstract_sax import AbstractSAX
from utils.utils import interpolate_segments


class SAX(AbstractSAX):
    """
    Symbolic Aggregate Approximation (SAX).

    :param alphabet_size: int (default = 3)
        The number of letters in the alphabet that shall be used for
        discretization. The alphabet starts from 'a' and ends with 'z' at the
        latest.
    :raises:
        ValueError: If the size of the alphabet is above 26 or below 1.

    References
    ----------
    [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel symbolic
    representation of time series. Data Mining and Knowledge Discovery,
    2007. vol. 15(107).
    """

    def __init__(self, alphabet_size=3):
        super().__init__(alphabet_size_avg=alphabet_size)

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

        # index i satisfies: breakpoints_avg[i-1] <= paa_value < breakpoints_avg[i]
        alphabet_idx = np.searchsorted(self.breakpoints_avg, df_paa, side="right")
        df_sax = pd.DataFrame(data=self.alphabet_avg[alphabet_idx],
                              index=df_paa.index, columns=df_paa.columns)
        return df_sax

    def inv_transform(self, df_sax, ts_size, window_size, symbol_mapping):
        """
        Approximate the original time series dataset by transforming its SAX
        representations into a time series dataset with the same size by
        assigning each point the symbol value of its segment.

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

        df_mapped = symbol_mapping.get_mapped(df_sax, self.alphabet_avg, self.breakpoints_avg)
        df_inv = interpolate_segments(df_mapped, ts_size, window_size)
        return df_inv

    def _distance(self, df_alphabet_idx, ts_size, sax_idx):
        """
        Compute pairwise distances between the given SAX representation and all
        other remaining SAX representations.
        The computation is based on the formula for the 'MINDIST' between two
        SAX words given in [1].

        :param ts_size: int
            The size of the original time series.
        :param df_alphabet_idx: dataframe of shape (num_segments, num_ts)
            The SAX representations mapped on its alphabet indices.
        :param sax_idx: int
            The column number of the current SAX representation, respectively
            of its alphabetical indexes mapping in the given 'df_sax' dataframe.
        :return:
            pd.Series of shape (sax_idx+1,)
        """

        sax_repr = df_alphabet_idx.iloc[:, sax_idx]
        # only SAX representations ahead are needed, since SAX distance is
        # symmetric and other distances were already computed
        sax_compare = df_alphabet_idx.iloc[:, sax_idx+1:]

        sax_diff = sax_compare.sub(sax_repr, axis=0)
        # implements first case of distances between single SAX symbols
        # use NaN values in df_abs to indicate resulting value of 0
        df_abs = sax_compare[abs(sax_diff) > 1]

        sax_repr = sax_repr.to_numpy()
        sax_repr = sax_repr.reshape((sax_repr.shape[0], 1))
        # implements second case of distances between single SAX symbols
        # do not consider NaN values, since they will be set to 0
        df_max_idx = np.maximum(df_abs, sax_repr) - 1
        df_min_idx = np.minimum(df_abs, sax_repr)

        mapping = dict(zip(range(self.breakpoints_avg.size), self.breakpoints_avg))
        df_max_idx.replace(to_replace=mapping, inplace=True)
        df_min_idx.replace(to_replace=mapping, inplace=True)
        df_diff_breakpts = df_max_idx - df_min_idx

        # contains symbol-wise distances between current SAX representation
        # and all other remaining SAX representations
        df_symb_distances = df_diff_breakpts.fillna(0)
        # implements actual SAX distance (MINDIST)
        squared_sums = df_symb_distances.pow(2).sum()
        num_segments = df_alphabet_idx.shape[0]
        sax_distances = np.sqrt((ts_size / num_segments) * squared_sums)

        return sax_distances

    def distance(self, df_sax, ts_size):
        """
        Compute pairwise distances between SAX representations.

        :param df_sax: dataframe of shape (num_segments, num_ts)
            The SAX representation of the time series dataset.
        :param ts_size: int
            The size of the original time series.
        :return: dataframe of shape (num_ts, num_ts)
            The returned dataframe is symmetric with zeros on the main diagonal,
            since the SAX distance is symmetric and positive definite.
        """

        if df_sax.shape[1] <= 1:
            raise ValueError("For pairwise distance computation, at least"
                             "two SAX representations need to be given.")

        mapping = dict(zip(self.alphabet_avg, range(self.alphabet_size_avg)))
        df_alphabet_idx = df_sax.replace(to_replace=mapping)
        df_sax_distances = pd.DataFrame(data=0, index=df_sax.columns, columns=df_sax.columns)
        num_ts = df_sax.shape[1]

        # last SAX representation has already been compared to all others
        for sax_idx in range(num_ts-1):
            # distances between current SAX representation and all other remaining SAX representations
            sax_distances = self._distance(df_alphabet_idx, ts_size, sax_idx)
            # use symmetry of resulting dataframe by building up a lower triangular matrix
            df_sax_distances.iloc[sax_idx+1:, sax_idx] = sax_distances

        # all values on main diagonal are zero due to initialization
        return df_sax_distances + df_sax_distances.T


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
