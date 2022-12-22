import unittest
import numpy as np

from approximation.paa import PAA


class TestPAA(unittest.TestCase):

    def test_one_univariate_time_series_in_2d_array_input(self):
        paa = PAA(window_size=2)
        ts = np.array([[1, 2, 3, 4, 5]])
        transformed = paa.transform(ts)
        # transformed: [[[1.5],[3.5]]]
        self.assertCountEqual(transformed[0, 0], [1.5])
        self.assertCountEqual(transformed[0, 1], [3.5])

    def test_one_univariate_time_series_in_3d_array_input(self):
        paa = PAA(window_size=2)
        ts = np.array([[[1], [2], [3], [4], [5]]])
        transformed = paa.transform(ts)
        # transformed: [[[1.5],[3.5]]]
        self.assertCountEqual(transformed[0, 0], [1.5])
        self.assertCountEqual(transformed[0, 1], [3.5])

    def test_multiple_univariate_time_series_in_2d_array_input(self):
        paa = PAA(window_size=2)
        ts = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        transformed = paa.transform(ts)
        # transformed: [[[1.5],[3.5]],
        #               [[6.5],[8.5]]]
        self.assertCountEqual(transformed[0, 0], [1.5])
        self.assertCountEqual(transformed[0, 1], [3.5])
        self.assertCountEqual(transformed[1, 0], [6.5])
        self.assertCountEqual(transformed[1, 1], [8.5])

    def test_multiple_univariate_time_series_in_3d_array_input(self):
        paa = PAA(window_size=2)
        ts = np.array([[[1], [2], [3], [4], [5]],
                       [[6], [7], [8], [9], [10]]])
        transformed = paa.transform(ts)
        # transformed: [[[1.5],[3.5]],
        #               [[6.5],[8.5]]]
        self.assertCountEqual(transformed[0, 0], [1.5])
        self.assertCountEqual(transformed[0, 1], [3.5])
        self.assertCountEqual(transformed[1, 0], [6.5])
        self.assertCountEqual(transformed[1, 1], [8.5])

    def test_multiple_multivariate_time_series_in_3d_array_input(self):
        paa = PAA(window_size=2)
        ts1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        ts2 = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
        X = np.stack([ts1, ts2], axis=0)
        # stacked: [[[1,2],[3,4],[5,6],[7,8]],
        #            [10,20],[30,40],[50,60],[70,80]]]
        transformed = paa.transform(X)
        self.assertCountEqual(transformed[0, 0], [2, 3])
        self.assertCountEqual(transformed[0, 1], [6, 7])
        self.assertCountEqual(transformed[1, 0], [20, 30])
        self.assertCountEqual(transformed[1, 1], [60, 70])
