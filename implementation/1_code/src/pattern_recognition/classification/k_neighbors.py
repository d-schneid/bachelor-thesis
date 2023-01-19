from sklearn.neighbors import KNeighborsClassifier

from utils import interpolate_segments
from pattern_recognition.utils import _get_linearized_encoded_sax


class KNeighborsTimeSeriesClassifier(KNeighborsClassifier):

    def __init__(self, n_neighbors=1, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski"):
        super().__init__(n_neighbors=n_neighbors, weights=weights,
                         algorithm=algorithm, leaf_size=leaf_size, p=p,
                         metric=metric)

    def fit_discretized(self, X, y, window_size, sax_variant):
        df_sax_linearized_encoded, df_sax = _get_linearized_encoded_sax(X, window_size, sax_variant)
        df_sax_linearized_encoded_interpolated = interpolate_segments(df_sax_linearized_encoded, X.shape[0], window_size)
        return super().fit(df_sax_linearized_encoded_interpolated.T, y)

    def fit(self, X, y):
        return super().fit(X.T, y)

    def score(self, X, y, sample_weight=None):
        return super().score(X.T, y, sample_weight)
