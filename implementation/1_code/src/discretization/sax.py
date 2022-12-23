import numpy as np
import pandas as pd
from scipy.stats import norm


class SAX:

    # expects time series to be normalized and in PAA representation

    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = np.array([chr(letter) for letter in
                                  range(ord('a'),ord('a') + self.alphabet_size)])
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size+1)[1:-1])

    def transform(self, df_paa):
        df_sax = pd.DataFrame(data=self.alphabet[np.searchsorted(
                                                            self.breakpoints,
                                                            df_paa,
                                                            side="right")],
                              index=df_paa.index, columns=df_paa.columns)
        return df_sax
