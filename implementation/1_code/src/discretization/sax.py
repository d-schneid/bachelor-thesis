import numpy as np
import pandas as pd
from scipy.stats import norm


class SAX:
    # expects time series to be normalized and in PAA representation

    # TODO: inherit from something like BaseApproximator

    # TODO: default value for alphabet size
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        letters = [chr(letter) for letter
                   in range(ord('a'), ord('a') + self.alphabet_size)]
        self.alphabet = np.array(letters)
        # TODO: super parent class

        probs = np.linspace(0, 1, self.alphabet_size+1)[1:-1]
        # quantiles of standard normal distribution
        self.breakpoints = norm.ppf(probs)

    def transform(self, df_paa):
        # index i satisfies: breakpoints[i-1] <= paa_value < breakpoints[i]
        alphabet_idx = np.searchsorted(self.breakpoints, df_paa, side="right")
        df_sax = pd.DataFrame(data=self.alphabet[alphabet_idx],
                              index=df_paa.index, columns=df_paa.columns)
        return df_sax

    # TODO: def inverse_transform