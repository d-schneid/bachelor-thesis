from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from approximation.paa import PAA
from pattern_recognition.utils import get_linearized_encoded_sax


class TimeSeriesClassifierMixin:

    def fit_discretized_encoded(self, df_norm, class_labels, window_size, sax_variant):
        """
        Fit the respective classifier from the discretized and encoded time
        series training dataset.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series training dataset that shall be discretized and
            encoded in order to fit the classifier from it.
        :param class_labels: pd.Series of shape (num_ts,)
            The class label of each time series from the given time series
            training dataset.
        :param window_size: int
            The size of the window that shall be used to transform the given
            time series training dataset into its PAA representation.
        :param sax_variant: AbstractSAX
            The SAX variant that shall be used to transform the PAA
            representation of the given time series training dataset into its
            symbolic representation (discretized representation).
        :return:
            The respective fitted classifier.
        """

        df_sax_linearized_encoded, df_sax = get_linearized_encoded_sax(df_norm, window_size, sax_variant)
        # transpose to align with sklearn
        return super().fit(df_sax_linearized_encoded.T, class_labels)

    def fit_inverse_transformed(self, df_norm, class_labels, window_size,
                                sax_variant, df_breakpoints=None, **symbol_mapping):
        """
        Fit the respective classifier from the inverse transformed time series
        training dataset.

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
            Can only be used for the aSAX as given 'sax_variant' and is ignored
            for other SAX variants.
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
        :return:
            The respective fitted classifier.
        """

        paa = PAA(window_size=window_size)
        df_paa = paa.transform(df_norm)
        df_inv_sax = sax_variant.transform_inv_transform(df_paa, df_norm, window_size,
                                                         df_breakpoints, **symbol_mapping)
        # transpose to align with sklearn
        return super().fit(df_inv_sax.T, class_labels)

    def fit(self, df_norm, class_labels):
        """
        Fit the respective classifier from the original normalized time series
        training dataset.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series training dataset the respective
            classifier shall be fit from.
        :param class_labels: pd.Series of shape (num_ts,)
            The class labels for the given time series.
        :return:
            The respective fitted classifier.
        """

        # transpose to align with sklearn
        return super().fit(df_norm.T, class_labels)

    def eval(self, df_norm, class_labels, eval_metrics_lst, average="macro"):
        """
        Compute evaluation scores of the fitted classifier based on the true
        class labels (ground truth).

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series dataset whose classes shall be predicted for each
            time seris by the fitted classifier.
        :param class_labels: pd.Series of shape (num_ts,)
            The true class labels (ground truth) of the given time series.
        :param eval_metrics_lst: list with elements of type ClassificationMetric
            Contains the classification evaluation metrics that shall be
            computed.
        :param average: {'micro', 'macro', 'samples', 'weighted', 'binary'} (default = 'macro')
            The type of averaging performed on the data by the respective
            classification metric.
            See documentation of classification metrics in 'sklearn' for
            further details.
        :return: dict
            Keys are the abbreviated names of the given classification
            evaluation metrics. Values are the corresponding scores of the
            classification evaluation metrics.
        """

        # transpose to align with sklearn
        predicted_labels = self.predict(df_norm.T)
        scores = {}
        for metric in eval_metrics_lst:
            score = metric.compute(class_labels, predicted_labels, average=average)
            scores[metric.abbreviation] = score
        return scores


class KNeighborsTimeSeriesClassifier(TimeSeriesClassifierMixin, KNeighborsClassifier):
    """
    A wrapper of the 'KNeighborsClassifier' of 'sklearn'.
    See 'sklearn' documentation for further details.
    """

    def __init__(self, n_neighbors=1, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski"):

        super().__init__(n_neighbors=n_neighbors, weights=weights,
                         algorithm=algorithm, leaf_size=leaf_size, p=p,
                         metric=metric)


class MLPTimeSeriesClassifier(TimeSeriesClassifierMixin, MLPClassifier):
    """
    A wrapper of the 'MLPClassifier' of 'sklearn'.
    See 'sklearn' documentation for further details.
    """

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 batch_size="auto", learning_rate="constant", learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, max_fun=15000):

        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver=solver, batch_size=batch_size, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init, power_t=power_t,
                         max_iter=max_iter, shuffle=shuffle, random_state=random_state,
                         momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                         early_stopping=early_stopping, validation_fraction=validation_fraction,
                         beta_1=beta_1, beta_2=beta_2, max_fun=max_fun)


class DecisionTreeTimeSeriesClassifier(TimeSeriesClassifierMixin, DecisionTreeClassifier):
    """
    A wrapper of the 'DecisionTreeClassifier' of 'sklearn'.
    See 'sklearn' documentation for further details.
    """

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):

        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
                         ccp_alpha=ccp_alpha)
