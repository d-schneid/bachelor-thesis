from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, jaccard_score)


"""
The following evaluation metrics for classification algorithms are listed in [1].

References
----------
[1] https://scikit-learn.org/stable/modules/model_evaluation.html section
3.3.1.1 'Common cases: predefined values'
"""


# the name of the module that shall be imported by the factory method
# 'get_metric_instance' located in 'utils'
CLASSIFICATION_METRICS_MODULE = "pattern_recognition.classification.eval_metrics"


class ClassificationMetric(ABC):
    """
    The abstract class from that all metrics that evaluate a classifier inherit.

    :param abbreviation: str
        The short name (abbreviation) of this metric.
    """

    def __init__(self, abbreviation):
        self.abbreviation = abbreviation

    @abstractmethod
    def compute(self, class_labels, predicted_labels, **kwargs):
        """
        Compute the score of this metric based on the ground truth class labels
        and the predicted class labels by the fitted classifier.

        :param class_labels: pd.Series of shape (num_ts,)
            The ground truth class labels of each time series.
        :param predicted_labels: pd.Series of shape (num_ts,)
            The predicted class labels of each time series by the fitted
            classifier.
        :param kwargs:
            Especially for determining the type of averaging performed on the
            data.
            See documentation of classification metrics in 'sklearn' for
            further details.
        :return: float
        """

        pass


class Accuracy(ClassificationMetric):
    """
    Compute the accuracy classification score of a fitted classifier.
    """

    def __init__(self):
        super().__init__("Accuracy")

    def compute(self, class_labels, predicted_labels, **kwargs):
        return accuracy_score(class_labels, predicted_labels)


class BalancedAccuracy(ClassificationMetric):
    """
    Compute the balanced accuracy classification score of a fitted classifier.
    The balanced accuracy accounts for imbalanced datasets. It is defined as
    the average of recall obtained on each class.
    """

    def __init__(self):
        super().__init__("bAccuracy")

    def compute(self, class_labels, predicted_labels, **kwargs):
        return balanced_accuracy_score(class_labels, predicted_labels)


class F1(ClassificationMetric):
    """
    Compute the F1 score of a fitted classifier.
    """

    def __init__(self):
        super().__init__("F1")

    def compute(self, class_labels, predicted_labels, **kwargs):
        return f1_score(class_labels, predicted_labels, **kwargs)


class Precision(ClassificationMetric):
    """
    Compute the precision of a fitted classifier.
    """

    def __init__(self):
        super().__init__("Precision")

    def compute(self, class_labels, predicted_labels, **kwargs):
        return precision_score(class_labels, predicted_labels, **kwargs)


class Recall(ClassificationMetric):
    """
    Compute the recall of a fitted classifier.
    """

    def __init__(self):
        super().__init__("Recall")

    def compute(self, class_labels, predicted_labels, **kwargs):
        return recall_score(class_labels, predicted_labels, **kwargs)


class JaccardSimilarityCoefficient(ClassificationMetric):
    """
    Compute the Jaccard similarity coefficient score.
    """

    def __init__(self):
        super().__init__("Jaccard")

    def compute(self, class_labels, predicted_labels, **kwargs):
        return jaccard_score(class_labels, predicted_labels, **kwargs)
