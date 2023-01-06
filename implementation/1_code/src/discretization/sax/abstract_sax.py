import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm


# the number of letters in the Latin alphabet
NUM_ALPHABET_LETTERS = 26


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


class AbstractSAX(ABC):
    """
    The abstract class from that all SAX variants (SAX, 1d-SAX, aSAX, eSAX)
    inherit.

    :param alphabet_size: int (default = 3)
        The number of letters in the alphabet that shall be used for
        discretization. The alphabet starts from 'a' and ends with 'z' at the
        latest.
    :raises:
        ValueError: If the size of the alphabet is above 26 or below 1.
    """

    def __init__(self, alphabet_size=3):
        if alphabet_size > NUM_ALPHABET_LETTERS or alphabet_size < 1:
            raise ValueError(f"The size of an alphabet needs to be between "
                             f"1 (inclusive) and {NUM_ALPHABET_LETTERS} (inclusive)")
        self.alphabet_size = alphabet_size
        letters = [chr(letter) for letter
                   in range(ord('a'), ord('a') + self.alphabet_size)]
        self.alphabet = np.array(letters)
        self.breakpoints = breakpoints(self.alphabet_size)

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
