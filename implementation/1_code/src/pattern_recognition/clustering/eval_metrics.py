from abc import ABC, abstractmethod
from sklearn.metrics import (
    rand_score, adjusted_rand_score, mutual_info_score,
    adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    fowlkes_mallows_score, silhouette_score, calinski_harabasz_score,
    davies_bouldin_score)


"""
The following evaluation metrics for clustering algorithms are described in [1].

References
----------
[1] https://scikit-learn.org/stable/modules/clustering.html section 2.3.10
'Clustering performance evaluation'
"""


# the name of the module that shall be imported by the factory method
# 'get_metric_instance' located in 'utils'
CLUSTERING_METRICS_MODULE = "pattern_recognition.clustering.eval_metrics"


class ClusteringMetric(ABC):
    """
    The abstract class from that all clustering metrics (ground truth and
    internal clustering metrics) inherit.

    :param abbreviation: str
        The short name (abbreviation) of this metric.
    """

    def __init__(self, abbreviation):
        self.abbreviation = abbreviation

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Compute the score of this metric.

        :param args:
            Non-keyword arguments (potentially) used by subclasses.
        :param kwargs:
            Keyword arguments (potentially) used by subclasses.
        :return: float
        """

        pass


class GroundTruthClusteringMetric(ClusteringMetric):
    """
    The abstract class from that all metrics that evaluate a clustering based
    on ground truth inherit.

    :param abbreviation: str
        The short name (abbreviation) of this metric.
    """

    def __init__(self, abbreviation):
        super().__init__(abbreviation)

    @abstractmethod
    def compute(self, cluster_labels, predicted_labels):
        """
        Compute the score of this metric based on the ground truth clustering
        and the fitted clustering.

        :param cluster_labels: pd.Series of shape (num_ts,)
            The cluster labels each given time series belongs to (ground truth).
        :param predicted_labels: np.array of shape (num_ts,)
            The cluster labels of each time series assigned by the respective
            clustering algorithm.
        :return: float
        """

        pass


class RandIndex(GroundTruthClusteringMetric):
    """
    Compute the score of the Rand index based on the ground truth clustering
    and the fitted clustering.
    """

    def __init__(self):
        super().__init__("Rand")

    def compute(self, cluster_labels, predicted_labels):
        return rand_score(cluster_labels, predicted_labels)


class AdjustedRandIndex(GroundTruthClusteringMetric):
    """
    Compute the score of the Adjusted Rand index based on the ground truth
    clustering and the fitted clustering.
    """

    def __init__(self):
        super().__init__("AdjRand")

    def compute(self, cluster_labels, predicted_labels):
        return adjusted_rand_score(cluster_labels, predicted_labels)


class MutualInformation(GroundTruthClusteringMetric):
    """
    Compute the Mutual Information score based on the ground truth clustering
    and the fitted clustering.
    """

    def __init__(self):
        super().__init__("MI")

    def compute(self, cluster_labels, predicted_labels):
        return mutual_info_score(cluster_labels, predicted_labels)


class AdjustedMutualInformation(GroundTruthClusteringMetric):
    """
    Compute the Adjusted Mutual Information score based on the ground truth
    clustering and the fitted clustering.
    """

    def __init__(self):
        super().__init__("AdjMI")

    def compute(self, cluster_labels, predicted_labels):
        return adjusted_mutual_info_score(cluster_labels, predicted_labels)


class NormalizedMutualInformation(GroundTruthClusteringMetric):
    """
    Compute the Normalized Mutual Information score based on the ground truth
    clustering and the fitted clustering.
    """

    def __init__(self):
        super().__init__("NormMI")

    def compute(self, cluster_labels, predicted_labels):
        return normalized_mutual_info_score(cluster_labels, predicted_labels)


class Homogeneity(GroundTruthClusteringMetric):
    """
    Compute the Homogeneity score based on the ground truth clustering and the
    fitted clustering.
    """

    def __init__(self):
        super().__init__("Homogeneity")

    def compute(self, cluster_labels, predicted_labels):
        return homogeneity_score(cluster_labels, predicted_labels)


class Completeness(GroundTruthClusteringMetric):
    """
    Compute the Completeness score based on the ground truth clustering and the
    fitted clustering.
    """

    def __init__(self):
        super().__init__("Completeness")

    def compute(self, cluster_labels, predicted_labels):
        return completeness_score(cluster_labels, predicted_labels)


class VMeasure(GroundTruthClusteringMetric):
    """
    Compute the V-measure score based on the ground truth clustering and the
    fitted clustering.
    """

    def __init__(self):
        super().__init__("V")

    def compute(self, cluster_labels, predicted_labels):
        return v_measure_score(cluster_labels, predicted_labels)


class FowlkesMallowsIndex(GroundTruthClusteringMetric):
    """
    Compute the score of the Fowlkes-Mallows index based on the ground truth
    clustering and the fitted clustering.
    """

    def __init__(self):
        super().__init__("Fowlkes-Mallows")

    def compute(self, cluster_labels, predicted_labels):
        return fowlkes_mallows_score(cluster_labels, predicted_labels)


class InternalClusteringMetric(ClusteringMetric):
    """
    The abstract class from that all metrics that evaluate a clustering based
    on internal characteristics (no ground truth) inherit.

    :param abbreviation: str
        The short name (abbreviation) of this metric.
    """

    def __init__(self, abbreviation):
        super().__init__(abbreviation)

    @abstractmethod
    def compute(self, df_norm, predicted_labels):
        """
        Compute the score of this metric based on internal characteristics (no
        ground truth) of the clustering.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series dataset that got clustered.
        :param predicted_labels: np.array of shape (num_ts,)
            The cluster labels of each time series assigned by the respective
            clustering algorithm.
        :return: float
        """

        pass


class SilhouetteCoefficient(InternalClusteringMetric):
    """
    Compute the score of the Silhouette Coefficient of the fitted clustering.
    """

    def __init__(self):
        super().__init__("Silhouette")

    def compute(self, df_norm, predicted_labels):
        return silhouette_score(df_norm, predicted_labels)


class CalinskiHarabaszIndex(InternalClusteringMetric):
    """
    Compute the score of the Calinski-Harabasz index of the fitted clustering.
    """

    def __init__(self):
        super().__init__("Calinski-Harabasz")

    def compute(self, df_norm, predicted_labels):
        return calinski_harabasz_score(df_norm, predicted_labels)


class DaviesBouldinIndex(InternalClusteringMetric):
    """
    Compute the score of the Davies-Bouldin index of the fitted clustering.
    """

    def __init__(self):
        super().__init__("Davies-Bouldin")

    def compute(self, df_norm, predicted_labels):
        return davies_bouldin_score(df_norm, predicted_labels)
