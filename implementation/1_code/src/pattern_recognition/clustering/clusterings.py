from sklearn.cluster import KMeans, Birch, AffinityPropagation

from approximation.paa import PAA
from pattern_recognition.clustering.eval_metrics import (
    GroundTruthClusteringMetric, InternalClusteringMetric)


class TimeSeriesClusteringMixin:

    def fit_discretized(self, df_norm, window_size, sax_variant,
                        df_breakpoints=None, **symbol_mapping):
        """
        Compute the respective clustering from the inverse transformed time
        series dataset.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series dataset that shall be inverse transformed and then
            clustered.
        :param window_size: int
            The size of the window that shall be used to transform the given
            time series dataset into its PAA representation.
        :param sax_variant: AbstractSAX
            The SAX variant that shall be used to transform the PAA
            representation of the given time series dataset into its symbolic
            representation.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            Can only be used for the aSAX as given 'sax_variant' and is ignored
            for other SAX variants.
            The individual breakpoints for the PAA representations of the given
            time series dataset that shall be used to transform the respective
            PAA representation into its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            This parameter is intended to allow breakpoints based on the
            k-means clustering of the original normalized time series data
            points.
        :param symbol_mapping: multiple objects of SymbolMapping
            The symbol mapping strategies that shall be used to inverse
            transform the symbolic representation of the given time series
            dataset.
            The appropriate symbol mapping strategies depend on the given
            'sax_variant'.
        :return:
            The respective fitted clustering estimator.
        """

        paa = PAA(window_size=window_size)
        df_paa = paa.transform(df_norm)
        df_inv_sax = sax_variant.transform_inv_transform(df_paa, df_norm, window_size,
                                                         df_breakpoints, **symbol_mapping)
        # transpose to align with sklearn
        return super().fit(df_inv_sax.T)

    def fit(self, df_norm):
        """
        Compute the respective clustering from the original normalized time
        series dataset.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset the respective
            clustering shall be computed from.
        :return:
            The respective fitted clustering estimator.
        """

        # transpose to align with sklearn
        return super().fit(df_norm.T)

    def predict(self, df_norm):
        """
        Predict the closest cluster each given time series belongs to.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset whose time series shall
            be predicted.
        :return: np.array of shape (num_ts,)
            Index of the cluster each time series belongs to.
        """

        # transpose to align with sklearn
        return super().predict(df_norm.T)

    def fit_predict(self, df_norm):
        """
        Compute the respective clustering and predict the closest cluster each
        given time series belongs to.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset whose time series shall
            be clustered and predicted.
        :return: np.array of shape (num_ts,)
            Index of the cluster each time series belongs to.
        """

        # transpose to align with sklearn
        return super().fit_predict(df_norm.T)

    def ground_truth_eval(self, df_norm, cluster_labels, eval_metrics_lst):
        """
        Compute evaluation scores of the fitted clustering based on the ground
        truth clustering.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset clusters shall be
            evaluated.
        :param cluster_labels: pd.Series of shape (num_ts,)
            The cluster labels each given time series belongs to (ground truth).
        :param eval_metrics_lst:
            list with elements of type GroundTruthClusteringMetric
                Contains the clustering evaluation metrics that shall be
                computed.
        :return: dict
            Keys are the abbreviated names of the given clustering evaluation
            metrics. Values are the corresponding scores of the clustering
            evaluation metrics.
        :raises:
            ValueError: If internal clustering metrics are used.
        """

        predicted_labels = self.predict(df_norm)
        scores = {}
        for metric in eval_metrics_lst:
            if isinstance(metric, InternalClusteringMetric):
                raise ValueError(f"{metric.abbreviation} is an internal "
                                 f"clustering metric which is not allowed "
                                 f"for this kind of evaluation.")
            score = metric.compute(cluster_labels, predicted_labels)
            scores[metric.abbreviation] = score
        return scores

    def internal_eval(self, df_norm, eval_metrics_lst):
        """
        Compute evaluation scores of the fitted clustering based on internal
        characteristics (no ground truth).

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series dataset that got clustered.
        :param eval_metrics_lst:
            list with elements of type InternalClusteringMetric
                Contains the clustering evaluation metrics that shall be
                computed.
        :return: dict
            Keys are the abbreviated names of the given clustering evaluation
            metrics. Values are the corresponding scores of the clustering
            evaluation metrics.
        :raises:
            ValueError: If ground truth clustering metrics are used.
        """

        scores = {}
        for metric in eval_metrics_lst:
            if isinstance(metric, GroundTruthClusteringMetric):
                raise ValueError(f"{metric.abbreviation} requires ground "
                                 f"truth which is not available for this kind "
                                 f"of evaluation.")
            score = metric.compute(df_norm.T, self.labels_)
            scores[metric.abbreviation] = score
        return scores


class TimeSeriesKMeans(TimeSeriesClusteringMixin, KMeans):
    """
    A wrapper of the 'KMeans' clustering algorithm of 'sklearn'.
    See 'sklearn' documentation for further details.
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300,
                 random_state=None, algorithm="lloyd"):

        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init,
                         max_iter=max_iter, random_state=random_state,
                         algorithm=algorithm)

    def internal_eval(self, df_norm, eval_metrics_lst):
        """
        In addition to the evaluation scores of the fitted clustering based on
        internal characteristics (no ground truth), add the SSQ error of the
        fitted clustering.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series dataset that got clustered.
        :param eval_metrics_lst:
            list with elements of type InternalClusteringMetric
                Contains the clustering evaluation metrics that shall be
                computed.
        :return: dict
            Keys are the abbreviated names of the given clustering evaluation
            metrics. Values are the scores of the corresponding clustering
            evaluation metrics.
            The dict contains the SSQ error, at least.
        :raises:
            ValueError: If ground truth clustering metrics are used.
        """

        scores = super().internal_eval(df_norm, eval_metrics_lst)
        scores["SSQ-Error"] = self.inertia_
        return scores


class TimeSeriesBirch(TimeSeriesClusteringMixin, Birch):
    """
    A wrapper of the 'Birch' clustering algorithm of 'sklearn'.
    See 'sklearn' documentation for further details.
    """

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):

        super().__init__(threshold=threshold, branching_factor=branching_factor,
                         n_clusters=n_clusters)


class TimeSeriesAffinityPropagation(TimeSeriesClusteringMixin, AffinityPropagation):
    """
    A wrapper of the 'AffinityPropagation' clustering algorithm of 'sklearn'.
    See 'sklearn' documentation for further details.
    """

    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15,
                 preference=None, affinity="euclidean", random_state=None):
        super().__init__(damping=damping, max_iter=max_iter,
                         convergence_iter=convergence_iter, preference=preference,
                         affinity=affinity, random_state=random_state)
