import inspect
import pandas as pd
from abc import ABC, abstractmethod
from sktime.performance_metrics.forecasting import (
    mean_absolute_error, mean_squared_error, median_absolute_error,
    median_squared_error, geometric_mean_absolute_error,
    geometric_mean_squared_error, mean_absolute_percentage_error,
    median_absolute_percentage_error, mean_squared_percentage_error,
    median_squared_percentage_error)


# the name of the module that shall be imported by the factory method
# 'get_pp_metric'
MODULE_NAME = "metrics.pp_metrics"


def get_pp_metric_instance(metric):
    """
    A factory method to create instances of subclasses of the abstract class
    'PPMetric'.

    :param metric: str
        The name of the subclass of the abstract class 'PPMetric' that shall be
        instantiated.
        Can be written in any way (lowercase, uppercase, ...), but all
        lowercase is preferred to avoid any dependency errors.
    :return:
        instance of class 'metric'
    """

    module = __import__(MODULE_NAME, fromlist=[metric])
    module_members = inspect.getmembers(module)
    # Find the member that has the same name as the given class (ignoring case)
    Metric = next(member for name, member in module_members
                  if name.lower() == metric.lower())

    # instantiate found subclass of abstract class 'PPMetric'
    return Metric()


class PPMetric(ABC):
    """
    The abstract class from that all metrics inherit.
    This abstract class is supposed to implement the 'template method' design
    pattern. In that way, existing metrics can be changed easily and new
    metrics can be added easily.

    :param abbreviation: str
        The short name (abbreviation) of this metric.
    """

    def __init__(self, abbreviation):
        self.abbreviation = abbreviation

    @abstractmethod
    def _compute(self, ts_norm, ts_inv):
        """
        The hook method that shall be overridden by subclasses.
        Determine what metric shall be used and compute the corresponding score
        for the given pair of time series based on that metric.

        :param ts_norm: pd.Series of shape (ts_size,)
            The original normalized time series.
        :param ts_inv: pd.Series of shape (ts_size,)
            The inverse transformed time series of the given original
            normalized time series
        :return:
            float
        """

        pass

    def compute(self, df_norm, df_inv):
        """
        Compute the score of this metric for each given pair of time series.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset.
        :param df_inv: dataframe of shape (ts_size, num_ts)
            The inverse transformed time series dataset of the given original
            normalized time series dataset.
        :return:
            pd.Series of shape (num_ts,)
        """

        results = []
        for i in range(df_norm.shape[1]):
            result = self._compute(df_norm.iloc[:, i], df_inv.iloc[:, i])
            results.append(result)
        return pd.Series(results)


class MeanAbsoluteError(PPMetric):
    """
    Compute the Mean Absolute Error between two time series as the mean of the
    absolute pointwise distances.
    """

    def __init__(self):
        super().__init__("MAE")

    def _compute(self, ts_norm, ts_inv):
        return mean_absolute_error(ts_norm, ts_inv)


class MeanSquaredError(PPMetric):
    """
    Compute the Mean Squared Error between two time series as the mean of the
    squared pointwise distances.
    """

    def __init__(self):
        super().__init__("MSE")

    def _compute(self, ts_norm, ts_inv):
        return mean_squared_error(ts_norm, ts_inv)


class RootMeanSquaredError(PPMetric):
    """
    Compute the Root Mean Squared Error between two time series as the square
    root of the mean of the squared pointwise distances (i.e. the square root
    of the Mean Squared Error).
    """

    def __init__(self):
        super().__init__("RMSE")

    def _compute(self, ts_norm, ts_inv):
        return mean_squared_error(ts_norm, ts_inv, square_root=True)


class MedianAbsoluteError(PPMetric):
    """
    Compute the Median Absolute Error between two time series as the median of
    the absolute pointwise distances.
    """

    def __init__(self):
        super().__init__("MedAE")

    def _compute(self, ts_norm, ts_inv):
        return median_absolute_error(ts_norm, ts_inv)


class MedianSquaredError(PPMetric):
    """
    Compute the Median Squared Error between two time series as the median of
    the squared pointwise distances.
    """

    def __init__(self):
        super().__init__("MedSE")

    def _compute(self, ts_norm, ts_inv):
        return median_squared_error(ts_norm, ts_inv)


class RootMedianSquaredError(PPMetric):
    """
    Compute the Root Median Squared Error between two time series as the square
    root of the median of the squared pointwise distances (i.e. the square root
    of the Median Squared Error).
    """

    def __init__(self):
        super().__init__("RMedSE")

    def _compute(self, ts_norm, ts_inv):
        return median_squared_error(ts_norm, ts_inv, square_root=True)


class GeometricMeanAbsoluteError(PPMetric):
    """
    Compute the Geometric Mean Absolute Error between two time series as the
    ts_size-th root of the product of absolute pointwise distances.
    """

    def __init__(self):
        super().__init__("GMAE")

    def _compute(self, ts_norm, ts_inv):
        return geometric_mean_absolute_error(ts_norm, ts_inv)


class GeometricMeanSquaredError(PPMetric):
    """
    Compute the Geometric Mean Squared Error between two time series as the
    ts_size-th root of the product of squared pointwise distances.
    """

    def __init__(self):
        super().__init__("GMSE")

    def _compute(self, ts_norm, ts_inv):
        return geometric_mean_squared_error(ts_norm, ts_inv)


class RootGeometricMeanSquaredError(PPMetric):
    """
    Compute the Root Geometric Mean Squared Error between two time series as
    the square root of the ts_size-th root of the product of squared pointwise
    distances (i.e. the square root of the Geometric Mean Squared Error).
    """

    def __init__(self):
        super().__init__("RGMSE")

    def _compute(self, ts_norm, ts_inv):
        return geometric_mean_squared_error(ts_norm, ts_inv, square_root=True)


class MeanAbsolutePercentageError(PPMetric):
    """
    Compute the Mean Absolute Percentage Error between two time series as the
    mean of the absolute relative pointwise distances (compared to the original
    points).
    """

    def __init__(self):
        super().__init__("MAPE")

    def _compute(self, ts_norm, ts_inv):
        return mean_absolute_percentage_error(ts_norm, ts_inv)


class SymmetricMeanAbsolutePercentageError(PPMetric):
    """
    Compute the Symmetric Mean Absolute Percentage Error between two time
    series as the mean of the absolute relative pointwise distances (compared
    to the mean of the two respective points).
    This metric is bounded: 0 <= metric <= 2.
    """

    def __init__(self):
        super().__init__("SMAPE")

    def _compute(self, ts_norm, ts_inv):
        return mean_absolute_percentage_error(ts_norm, ts_inv, symmetric=True)


class MedianAbsolutePercentageError(PPMetric):
    """
    Compute the Median Absolute Percentage Error between two time series as the
    median of the absolute relative pointwise distances (compared to the
    original points).
    """

    def __init__(self):
        super().__init__("MedAPE")

    def _compute(self, ts_norm, ts_inv):
        return median_absolute_percentage_error(ts_norm, ts_inv)


class SymmetricMedianAbsolutePercentageError(PPMetric):
    """
    Compute the Symmetric Median Absolute Percentage Error between two time
    series as the median of the absolute relative pointwise distances (compared
    to the mean of the two respective points).
    """

    def __init__(self):
        super().__init__("SMedAPE")

    def _compute(self, ts_norm, ts_inv):
        return median_absolute_percentage_error(ts_norm, ts_inv, symmetric=True)


class MeanSquaredPercentageError(PPMetric):
    """
    Compute the Mean Squared Percentage Error between two time series as the
    mean of the squared relative pointwise distances (compared to the original
    points).
    """

    def __init__(self):
        super().__init__("MSPE")

    def _compute(self, ts_norm, ts_inv):
        return mean_squared_percentage_error(ts_norm, ts_inv)


class RootMeanSquaredPercentageError(PPMetric):
    """
    Compute the Root Mean Squared Percentage Error between two time series as
    the square root of the mean of the squared relative pointwise distances
    (compared to the original points) (i.e. the square root of the Mean Squared
    Percentage Error).
    """

    def __init__(self):
        super().__init__("RMSPE")

    def _compute(self, ts_norm, ts_inv):
        return mean_squared_percentage_error(ts_norm, ts_inv, square_root=True)


class SymmetricMeanSquaredPercentageError(PPMetric):
    """
    Compute the Symmetric Mean Squared Percentage Error between two time series
    as the mean of the squared relative pointwise distances (compared to the
    mean of the two respective points).
    """

    def __init__(self):
        super().__init__("SMSPE")

    def _compute(self, ts_norm, ts_inv):
        return mean_squared_percentage_error(ts_norm, ts_inv, symmetric=True)


class RootSymmetricMeanSquaredPercentageError(PPMetric):
    """
    Compute the Root Symmetric Mean Squared Percentage Error between two time
    series as the square root of the mean of the squared relative pointwise
    distances (compared to the mean of the two respective points) (i.e. the
    square root of the Symmetric Mean Squared Percentage Error).
    """

    def __init__(self):
        super().__init__("RSMSPE")

    def _compute(self, ts_norm, ts_inv):
        return mean_squared_percentage_error(ts_norm, ts_inv, square_root=True, symmetric=True)


class MedianSquaredPercentageError(PPMetric):
    """
    Compute the Median Squared Percentage Error between two time series as the
    median of the squared relative pointwise distances (compared to the
    original points).
    """

    def __init__(self):
        super().__init__("MedSPE")

    def _compute(self, ts_norm, ts_inv):
        return median_squared_percentage_error(ts_norm, ts_inv)


class RootMedianSquaredPercentageError(PPMetric):
    """
    Compute the Root Median Squared Percentage Error between two time series as
    the square root of the median of the squared relative pointwise distances
    (compared to the original points) (i.e. the square root of the Median
    Squared Percentage Error).
    """

    def __init__(self):
        super().__init__("RMedSPE")

    def _compute(self, ts_norm, ts_inv):
        return median_squared_percentage_error(ts_norm, ts_inv, square_root=True)


class SymmetricMedianSquaredPercentageError(PPMetric):
    """
    Compute the Symmetric Median Squared Percentage Error between two time
    series as the median of the squared relative pointwise distances (compared
    to the mean of the two respective points).
    """

    def __init__(self):
        super().__init__("SMedSPE")

    def _compute(self, ts_norm, ts_inv):
        return median_squared_percentage_error(ts_norm, ts_inv, symmetric=True)


class RootSymmetricMedianSquaredPercentageError(PPMetric):
    """
    Compute the Root Symmetric Median Squared Percentage Error between two time
    series as the square root of the median of the squared relative pointwise
    distances (compared to the mean of the two respective points) (i.e. the
    square root of the Symmetric Median Squared Percentage Error).
    """

    def __init__(self):
        super().__init__("RSMedSPE")

    def _compute(self, ts_norm, ts_inv):
        return median_squared_percentage_error(ts_norm, ts_inv, square_root=True, symmetric=True)
