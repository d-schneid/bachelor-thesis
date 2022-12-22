import numpy as np
from sklearn.utils.validation import check_array
from tslearn.utils.utils import check_dims

from utils.utils import constant_segmentation


class PAA:
    # TODO: inherit from something like BaseApproximator

    def __init__(self, window_size=1):
        self.window_size = window_size
        # TODO: super parent class

    # TODO: schauen, ob parallelisierbar
    def _transform(self, X):
        num_ts, ts_size, dim = X.shape
        start, end, num_segments = constant_segmentation(ts_size, self.window_size)
        X_paa = np.empty((num_ts, num_segments, dim))
        for i_ts in range(num_ts):
            for i_seg in range(num_segments):
                segment = X[i_ts, start[i_seg]:end[i_seg], :]
                X_paa[i_ts, i_seg, :] = segment.mean(axis=0)
        return X_paa

    def transform(self, X):
        """
        Reduce the dimensionality of each time series by transforming it into
        its PAA representation.

        :param X: array-like of shape (num_ts, ts_size, dim) or (num_ts, ts_size)
            The time series dataset.
            If shape is (num_ts, ts_size), X is reshaped to a 3-dimensional
            array of X.shape[0] univariate time series of length X.shape[1].
        :return:
            numpy.ndarray of shape (num_ts, num_segments, dim)
        """

        X = check_array(X, allow_nd=True)
        X = check_dims(X)
        return self._transform(X)
