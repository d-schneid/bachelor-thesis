import unittest
import pandas as pd

from approximation.paa import PAA


class TestPAA(unittest.TestCase):

    def test_univariate_time_series(self):
        paa = PAA(window_size=2)
        ts = pd.DataFrame([1, 2, 3, 4, 5])
        transformed = paa.transform(ts)
        self.assertEqual(transformed.iloc[0, 0], 1.5)
        self.assertEqual(transformed.iloc[1, 0], 3.5)

    def test_multivariate_time_series(self):
        paa = PAA(window_size=2)
        ts = pd.DataFrame([[1, 6],
                           [2, 7],
                           [3, 8],
                           [4, 9],
                           [5, 10]])
        transformed = paa.transform(ts)
        self.assertEqual(transformed.iloc[0, 0], 1.5)
        self.assertEqual(transformed.iloc[1, 0], 3.5)
        self.assertEqual(transformed.iloc[0, 1], 6.5)
        self.assertEqual(transformed.iloc[1, 1], 8.5)
