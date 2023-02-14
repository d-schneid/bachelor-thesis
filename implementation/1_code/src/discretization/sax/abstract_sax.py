import pandas as pd
from scipy.stats import norm
from abc import ABC

from discretization.abstract_discretization import AbstractDiscretization


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


class AbstractSAX(AbstractDiscretization, ABC):
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
        super().__init__(alphabet_size=alphabet_size)
        self.breakpoints = breakpoints(self.alphabet_size)
