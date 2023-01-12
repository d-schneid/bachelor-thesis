import numpy as np
import pandas as pd

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
        super().__init__(alphabet_size=alphabet_size)

    def transform(self, df_paa, *args, **kwargs):
        """
        Transform the PAA representation of each time series into its SAX
        representation (i.e. assign each PAA representation its respective
        SAX word).
        It does not modify the given PAA representations before computation,
        such as normalization. Therefore, the modification of the PAA
        representations (e.g. normalization) is the responsibility of the user.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of a time series dataset that shall be
            transformed into their SAX representations.
        :return:
            dataframe of shape (num_segments, num_ts)
        """

        # index i satisfies: breakpoints[i-1] <= paa_value < breakpoints[i]
        alphabet_idx = np.searchsorted(self.breakpoints, df_paa, side="right")
        df_sax = pd.DataFrame(data=self.alphabet[alphabet_idx],
                              index=df_paa.index, columns=df_paa.columns)
        return df_sax

    def inv_transform(self, df_sax, ts_size, window_size, symbol_mapping,
                      *args, **kwargs):
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

        df_mapped = symbol_mapping.get_mapped(df_sax, self.alphabet, self.breakpoints)
        df_inv_sax = interpolate_segments(df_mapped, ts_size, window_size)
        return df_inv_sax

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

        mapping = dict(zip(range(self.breakpoints.size), self.breakpoints))
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

        mapping = dict(zip(self.alphabet, range(self.alphabet_size)))
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
