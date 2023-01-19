import numpy as np
import pandas as pd

from discretization.sax.abstract_sax import AbstractSAX
from utils import constant_segmentation, interpolate_segments


def _compute_segment_min_idx(segment):
    """
    Compute the minimum value of the given time series segment along with its
    corresponding index for each time series.

    :param segment: dataframe of shape (segment_size, num_ts)
        The time series segment in which the minimum value along with its
        corresponding index shall be computed for each time series.
    :return:
        pd.Series of shape (num_ts,)
        pd.Series of shape (num_ts,)
    """

    return segment.idxmin(), segment.min()


class ExtendedSAX(AbstractSAX):
    """
    Extended Symbolic Aggregate Approximation (eSAX).

    :param alphabet_size: int (default = 3)
        The number of letters in the alphabet that shall be used for
        discretization. The alphabet starts from 'a' and ends with 'z' at the
        latest.
    :raises:
        ValueError: If the size of the alphabet is above 26 or below 1.

    References
    ----------
    [1] Lkhagva, B., Suzuki, Y., & Kawagoe, K. (2006, April). New time series
    data representation ESAX for financial applications. In 22nd International
    Conference on Data Engineering Workshops (ICDEW'06) (pp. x115-x115). IEEE.
    """

    def __init__(self, alphabet_size=3):
        super().__init__(alphabet_size=alphabet_size)
        self.symbols_per_segment = 3

    def _transform(self, segment_list):
        """
        Transform the given segment values (mean, maximum, or minimum) into
        their SAX symbols.

        :param segment_list: list of len(segment_list) = num_segments
            Contains pd.Series that contain tuples of
            (index of segment value within original time series, segment value).
        :return: dataframe of shape (num_segments, num_ts)
            Each cell of this dataframe contains a list with one tuple of
            (index of segment value within original time series, symbol value of segment value).
        """

        # for each time series and in each segment (index, respective value)
        df = pd.concat(segment_list, axis=1).T
        # for each time series and in each segment (index, respective symbol)
        df_sax = df.applymap(lambda tup: (tup[0], self.alphabet[
            np.searchsorted(self.breakpoints, tup[1], side="right")]))
        return df_sax.applymap(lambda tup: [tup])

    def transform(self, df_paa, df_norm, window_size):
        """
        Transform each time series into its eSAX representation.
        It does not modify the given time series before computation, such as
        normalization. Therefore, the modification of the time series (e.g.
        normalization) is the responsibility of the user.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of the given normalized time series
            dataset.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The normalized time series dataset that shall be transformed into
            an eSAX representation.
        :param window_size: int
            The size of the segments with which the given PAA representations
            were created.
        :return:
            df_e_sax: dataframe of shape (num_segments, num_ts)
                The eSAX representation of each given time series.
            df_sax_mean: dataframe of shape (num_segments, num_ts)
                The SAX representation of each given PAA representation
                together with the index of the middle of each segment within
                the original time series. So, each cell of this dataframe
                contains a tuple of
                (index of segment middle, SAX symbol of segment mean).
            df_sax_max: dataframe of shape (num_segments, num_ts)
                The SAX representation of the maximum value within each segment
                together with its index within the original time series. So,
                each cell of this dataframe contains a tuple of
                (index of segment maximum, SAX symbol of segment maximum).
            df_sax_min: dataframe of shape (num_segments, num_ts)
                The SAX representation of the minimum value within each segment
                together with its index within the original time series. So,
                each cell of this dataframe contains tuples of
                (index of segment minimum, SAX symbol of segment minimum).
        """

        ts_size = df_norm.shape[0]
        start, end, num_segments = constant_segmentation(ts_size, window_size)
        t_segment_mid = pd.Series(data=(start + end - 1) / 2)

        segment_means, segment_maxs, segment_mins = [], [], []
        for i in range(num_segments):
            segment = df_norm.iloc[start[i]:end[i]]

            segment_mean = df_paa.iloc[i]
            pos_mid = pd.Series([t_segment_mid[i]] * segment_mean.size)
            segment_means.append(pd.Series(zip(pos_mid, segment_mean)))

            # find max with negative values
            pos_min, segment_min = _compute_segment_min_idx(-segment)
            segment_maxs.append(pd.Series(zip(pos_min, -segment_min)))

            pos_min, segment_min = _compute_segment_min_idx(segment)
            segment_mins.append(pd.Series(zip(pos_min, segment_min)))

        # transform second elements of tuples (values) into SAX symbols
        df_sax_mean = self._transform(segment_means)
        df_sax_max = self._transform(segment_maxs)
        df_sax_min = self._transform(segment_mins)

        # list concatenation of lists with one tuple (index, respective value)
        df_e_sax = df_sax_mean + df_sax_max + df_sax_min
        # sort lists according to index (i.e. position of respective values)
        df_e_sax = df_e_sax.applymap(lambda lst: sorted(lst, key=lambda tup: tup[0]))
        # symbols are now in correct order (i.e. increasing in position)
        df_e_sax = df_e_sax.applymap(lambda lst: ''.join([tup[1] for tup in lst]))

        # remove lists around tuples for easier further processing
        # (e.g. inverse transformation)
        df_sax_mean = df_sax_mean.applymap(lambda lst: lst[0])
        df_sax_max = df_sax_max.applymap(lambda lst: lst[0])
        df_sax_min = df_sax_min.applymap(lambda lst: lst[0])

        return df_e_sax, df_sax_mean, df_sax_max, df_sax_min

    def _inv_transform_extrema(self, df_sax_extrema, symbol_mapping_extrema):
        """
        Assign the respective symbol value for each SAX symbol of each segment
        extrema (minimum or maximum value).

        :param df_sax_extrema: dataframe of shape (num_segments, num_ts)
            Contains the segment extrema (maximum or minimum values) along with
            their index within the original time series for each time series.
            This is one of the outputs of the transformation of the time series
            dataset into its eSAX representations ('transform' method).
        :param symbol_mapping_extrema: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the SAX symbols of the segment extrema (maximum or minimum values).
        :return: dataframe of shape (num_segments, num_ts)
            Each cell of this dataframe contain a tuple of
            (index of segment extrema, symbol value of segment extrema).
        """

        df_symbols = df_sax_extrema.applymap(lambda tup: tup[1])
        df_mapped = symbol_mapping_extrema.get_mapped(df_symbols, self.alphabet, self.breakpoints)
        df_mapped = df_mapped.applymap(lambda mapped_val: (mapped_val,))
        df_pos = df_sax_extrema.applymap(lambda tup: (tup[0],))
        # tuple concatenation, so that the resulting dataframe contains tuples
        # (index, mapped value), where index is equal to the position within
        # the original time series
        return df_pos + df_mapped

    def inv_transform(self, df_sax_mean, df_sax_max, df_sax_min, ts_size, window_size,
                      symbol_mapping_mean, symbol_mapping_max=None, symbol_mapping_min=None):
        """
        Approximate the original time series dataset by transforming its eSAX
        representations into a time series dataset with the same size.
        Thereby, the symbols belonging to the segment mean, segment maximum,
        and segment minimum are mapped to their symbol values as in the classic
        SAX. Then, the points in a segment are all assigned the symbol value
        for the respective segment mean except for the maximum and minimum
        value of the segment. The maximum and minimum value of the segment are
        assigned the symbol value for the segment maximum and segment minimum,
        respectively.

        :param df_sax_mean: dataframe of shape (num_segments, num_ts)
            Contains the segment means along with the index of the segment
            middle within the original time series for each time series.
            This is one of the outputs of the transformation of the time series
            dataset into its eSAX representations ('transform' method).
        :param df_sax_max: dataframe of shape (num_segments, num_ts)
            Contains the segment maximum values along with their index within
            the original time series for each time series.
            This is one of the outputs of the transformation of the time series
            dataset into its eSAX representations ('transform' method).
        :param df_sax_min: dataframe of shape (num_segments, num_ts)
            Contains the segment minimum values along with their index within
            the original time series for each time series.
            This is one of the outputs of the transformation of the time series
            dataset into its eSAX representations ('transform' method).
        :param ts_size: int
            The size of the original time series
        :param window_size: int
            The size of the segments with which the given dataframes were
            created.
        :param symbol_mapping_mean: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the SAX symbols of the segment means.
        :param symbol_mapping_max: SymbolMapping (default = None)
            The symbol mapping strategy that determines the symbol values for
            the SAX symbols of the segment maximum values.
            If 'None', the symbol mapping strategy given by
            'symbol_mapping_mean' is used.
        :param symbol_mapping_min:
            The symbol mapping strategy that determines the symbol values for
            the SAX symbols of the segment minimum values.
            If 'None', the symbol mapping strategy given by
            'symbol_mapping_mean' is used.
        :return:
            dataframe of shape (ts_size, num_ts)
        """

        df_symbols_mean = df_sax_mean.applymap(lambda tup: tup[1])
        df_mapped_mean = symbol_mapping_mean.get_mapped(df_symbols_mean, self.alphabet, self.breakpoints)
        df_inv_mean = interpolate_segments(df_mapped_mean, ts_size, window_size)

        symbol_mapping_max = symbol_mapping_mean if symbol_mapping_max is None else symbol_mapping_max
        df_mapped_max = self._inv_transform_extrema(df_sax_max, symbol_mapping_max)
        symbol_mapping_min = symbol_mapping_mean if symbol_mapping_min is None else symbol_mapping_min
        df_mapped_min = self._inv_transform_extrema(df_sax_min, symbol_mapping_min)

        # create a dictionary for each time series where the key is the
        # position within the original time series and the value is the value
        # of the mapped maximum or minimum point
        mapping_max = [{tup[0]: tup[1] for tup in df_mapped_max.iloc[:, i]}
                       for i in range(df_mapped_max.shape[1])]
        mapping_min = [{tup[0]: tup[1] for tup in df_mapped_min.iloc[:, i]}
                       for i in range(df_mapped_min.shape[1])]

        df_inv_e_sax = df_inv_mean
        # set the mapped values of the minimum and maximum points for each time
        # series by overwriting the respective values of the segment mean
        # interpolation
        for i in range(df_inv_e_sax.shape[1]):
            df_inv_e_sax.iloc[list(mapping_max[i].keys()), i] = mapping_max[i]
            df_inv_e_sax.iloc[list(mapping_min[i].keys()), i] = mapping_min[i]

        return df_inv_e_sax

    def transform_inv_transform(self, df_paa, df_norm, window_size, df_breakpoints=None, **symbol_mapping):
        ts_size = df_norm.shape[0]
        df_e_sax, df_sax_mean, df_sax_max, df_sax_min = self.transform(df_paa, df_norm, window_size)
        return self.inv_transform(df_sax_mean, df_sax_max, df_sax_min, ts_size,
                                  window_size, **symbol_mapping)
