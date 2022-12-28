import numpy as np
from abc import ABC
from scipy.stats import norm


NUM_ALPHABET_LETTERS = 26


def breakpoints(alphabet_size, scale=1):
    quantiles = [numerator / alphabet_size for numerator in range(1, alphabet_size)]
    # z-values of quantiles of Gaussian distribution with variance 'scale'
    breakpts = norm.ppf(quantiles, scale=scale)
    return breakpts


class AbstractSAX(ABC):

    def __init__(self, alphabet_size_avg=3):
        if NUM_ALPHABET_LETTERS < alphabet_size_avg < 1:
            raise ValueError(f"The size of an alphabet needs to be between "
                             f"1 (inclusive) and {NUM_ALPHABET_LETTERS} (inclusive)")
        self.alphabet_size_avg = alphabet_size_avg
        letters_avg = [chr(letter) for letter
                       in range(ord('a'), ord('a') + self.alphabet_size_avg)]
        self.alphabet_avg = np.array(letters_avg)
        self.breakpoints_avg = breakpoints(self.alphabet_size_avg)
