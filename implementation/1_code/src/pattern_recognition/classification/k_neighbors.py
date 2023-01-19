from sklearn.neighbors import KNeighborsClassifier

from approximation.paa import PAA


# TODO: comment
class KNeighborsTimeSeriesClassifier(KNeighborsClassifier):

    def __init__(self, n_neighbors=1, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski"):
        super().__init__(n_neighbors=n_neighbors, weights=weights,
                         algorithm=algorithm, leaf_size=leaf_size, p=p,
                         metric=metric)

    def fit_discretized(self, df_norm, class_labels, window_size, sax_variant,
                        df_breakpoints=None, **symbol_mapping):
        """
        Fit the k-nearest neighbors classifier from the inverse transformed
        time series training dataset.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series training dataset that shall be inverse transformed
            in order to fit the classifier from it.
        :param class_labels: pd.Series of shape (num_ts,)
            The class label of each time series from the given time series
            training dataset.
        :param window_size: int
            The size of the window that shall be used to transform the given
            time series training dataset into its PAA representation.
        :param sax_variant: AbstractSAX
            The SAX variant that shall be used to transform the PAA
            representation of the given time series training dataset into its
            symbolic representation.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            Can only be used for the aSAX. Ignored for other SAX variants.
            The individual breakpoints for the PAA representations of the given
            time series training dataset that shall be used to transform the
            respective PAA representation into its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            This parameter is intended to allow breakpoints based on the
            k-means clustering of the original normalized time series data
            points.
        :param symbol_mapping: multiple objects of SymbolMapping
            The symbol mapping strategies that shall be used to inverse
            transform the symbolic representation of the given time series
            training dataset.
            The appropriate symbol mapping strategies depend on the given
            'sax_variant'.
        :return: KNeighborsTimeSeriesClassifier
            The fitted k-nearest neighbors classifier.
        """

        paa = PAA(window_size=window_size)
        df_paa = paa.transform(df_norm)
        df_inv_sax = sax_variant.transform_inv_transform(df_paa, df_norm, window_size,
                                                         df_breakpoints, **symbol_mapping)
        # transpose to align with sklearn
        return super().fit(df_inv_sax.T, class_labels)

    def fit(self, df_norm, class_labels):
        return super().fit(df_norm.T, class_labels)

    def score(self, df_norm, class_labels, sample_weight=None):
        return super().score(df_norm.T, class_labels, sample_weight)
