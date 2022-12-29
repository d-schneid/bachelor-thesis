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


# TODO: comment
class AbstractSAX(ABC):

    def __init__(self, alphabet_size_avg=3):
        if alphabet_size_avg > NUM_ALPHABET_LETTERS or alphabet_size_avg < 1:
            raise ValueError(f"The size of an alphabet needs to be between "
                             f"1 (inclusive) and {NUM_ALPHABET_LETTERS} (inclusive)")
        self.alphabet_size_avg = alphabet_size_avg
        letters_avg = [chr(letter) for letter
                       in range(ord('a'), ord('a') + self.alphabet_size_avg)]
        self.alphabet_avg = np.array(letters_avg)
        self.breakpoints_avg = breakpoints(self.alphabet_size_avg)

    # TODO: extraxct SAX transformation of PAA as it is done in both, SAX and 1d-SAX
    # but then, class is instantiable, because it does not have an abstract method
    @abstractmethod
    def transform(self, *args, **kwargs):
        pass
