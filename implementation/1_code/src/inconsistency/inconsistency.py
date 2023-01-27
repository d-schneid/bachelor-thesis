import random
import pandas as pd
from abc import ABC, abstractmethod

from approximation.paa import PAA


class InconsistencyMode(ABC):
    """
    The abstract class from that all inconsistency modes inherit. In its
    subclasses it is determined how the inconsistency metrics are computed.
    """

    @abstractmethod
    def compute_inconsistency(self, epsilon_pts_bins):
        pass

    @abstractmethod
    def compute_sum(self, epsilon_pts_bins):
        pass

    @abstractmethod
    def compute_num_bins(self, epsilon_pts_bins):
        pass


class Unweighted(InconsistencyMode):
    """
    This class shall be used to compute the mean inconsistency rate per time
    series point and the mean number of discretization bins per time series
    point.
    """

    def compute_inconsistency(self, epsilon_pts_bins):
        """
        Compute the inconsistency rate of the randomly chosen time series
        point.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the randomly
            chosen time series point with their corresponding bins.
        :return: float
        """

        inconsistency = epsilon_pts_bins.size - epsilon_pts_bins.value_counts().max()
        return inconsistency / epsilon_pts_bins.size

    def compute_sum(self, epsilon_pts_bins):
        """
        Helps to count the number of randomly chosen points.

        :param epsilon_pts_bins: pd.Series
            Is ignored, just here for API consistency.
            Contains all points in the epsilon neighborhood of the randomly
            chosen time series point with their corresponding bins.
        :return: int
        """

        return 1

    def compute_num_bins(self, epsilon_pts_bins):
        """
        Compute the number of different discretization bins the randomly chosen
        time series point was assigned to.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the randomly
            chosen time series point with their corresponding bins.
        :return: int
        """

        return len(epsilon_pts_bins.value_counts())


class Weighted(InconsistencyMode):
    """
    This class shall be used to compute the mean inconsistency rate per time
    series point weighted by the occurrences of the respective time series
    point in the time series and the mean number of discretization bins per
    time series point weighted by the occurrences of the respective time series
    point in the time series.
    """

    def compute_inconsistency(self, epsilon_pts_bins):
        """
        Compute the absolute inconsistency value of the randomly chosen time
        series point.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the randomly
            chosen time series point with their corresponding bins.
        :return: int
        """

        inconsistency = epsilon_pts_bins.size - epsilon_pts_bins.value_counts().max()
        return inconsistency

    def compute_sum(self, epsilon_pts_bins):
        """
        Helps to sum up the weights of the randomly chosen time series points.
        The weight of a randomly chosen time series point are the occurrences
        of this point in the time series.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the randomly
            chosen time series point with their corresponding bins.
        :return: int
        """

        return epsilon_pts_bins.size

    def compute_num_bins(self, epsilon_pts_bins):
        """
        Compute the number of different discretization bins the randomly chosen
        time series point was assigned to; weighted by the occurrences of the
        respective time series point in the time series

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the randomly
            chosen time series point with their corresponding bins.
        :return: int
        """

        return epsilon_pts_bins.size * len(epsilon_pts_bins.value_counts())


def compute_inconsistency_metrics(df_norm, window_size, sax_variant, inconsistency_mode,
                                  num_iter, epsilon, seed=1, df_breakpoints=None):
    """
    Compute the unweighted or weighted mean inconsistency rate per time series
    point and the unweighted or weighted mean number of discretization bins per
    time series point.
    The inconsistency metrics are computed based on a Monte Carlo simulation
    where in each round a point from the current time series is drawn at
    random and evaluated based on its inconsistency.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series for that the inconsistency metrics shall be computed.
    :param window_size: int
        The size of the window that shall be used to transform the given time
        series dataset into its PAA representations.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used for transforming the given time
        series dataset into its symbolic representations.
    :param inconsistency_mode: InconsistencyMode
        Determines if the unweighted or weighted inconsistency metrics shall be
        computed.
    :param num_iter: int
        For each given time series, the number of times a random point from the
        respective time series shall be chosen.
    :param epsilon: float
        The tolerance for time series points to be considered equal.
        All points that are in the neighborhood of the randomly chosen point:
        current_point - epsilon <= current_point <= current_point + epsilon
        are considered equal to this point.
        This is needed, because in most time series floating points are used.
    :param seed: int (default = 1)
        The seed that shall be used as a starting point to randomly choose
        a point from the current time series in the current iteration whose
        inconsistency metrics shall be computed.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
        Can only be used for the aSAX as 'sax_variant' and is ignored for other
        SAX variants.
        The individual breakpoints for the PAA representations of the given
        time series dataset that shall be used to transform the respective PAA
        representation into its aSAX representation.
        If None, the respective breakpoints resulting from the k-means
        clustering of the respective PAA points are used.
        This parameter is intended to allow breakpoints based on the
        k-means clustering of the original normalized time series data points.
    :return: tuple of len = 2
        Both entries contain a pd.Series of shape (num_ts,). The first entry
        contains the (weighted/unweighted) mean inconsistency rate per time
        series point. The second entry contains the (weighted/unweighted) mean
        number of discretization bins per time series point.
    """

    paa = PAA(window_size=window_size)
    df_paa = paa.transform(df_norm)
    # interpolate symbolic representation to all original points
    df_symbolic_ts = sax_variant.transform_to_symbolic_ts(df_paa, df_norm, window_size, df_breakpoints)
    # has only effect for 1d-SAX
    df_symbolic_ts = sax_variant.adjust_symbolic_ts(df_symbolic_ts)

    num_ts = df_norm.shape[1]
    total_inconsistency, total_sum, total_bins_per_pt = [0] * num_ts, [0] * num_ts, [0] * num_ts
    for i in range(num_ts):
        current_ts = df_norm.iloc[:, i]
        current_symbolic_ts = df_symbolic_ts.iloc[:, i]
        for it in range(num_iter):
            random.seed(seed + i + it)
            current_pt = random.choice(current_ts)
            lower_bound, upper_bound = current_pt - epsilon, current_pt + epsilon
            # epsilon neighborhood of current_pt
            epsilon_pts = current_ts[current_ts.between(lower_bound, upper_bound)]
            epsilon_pts_idxs = list(epsilon_pts.index)
            # Bins belonging to points in epsilon neighborhood of current_pt
            epsilon_pts_bins = current_symbolic_ts.iloc[epsilon_pts_idxs]
            # most common bin is seen as the "true" bin for the current_point
            # remaining points were assigned the "wrong" bin
            total_inconsistency[i] += inconsistency_mode.compute_inconsistency(epsilon_pts_bins)
            total_sum[i] += inconsistency_mode.compute_sum(epsilon_pts_bins)
            # current_point is assigned to multiple bins
            total_bins_per_pt[i] += inconsistency_mode.compute_num_bins(epsilon_pts_bins)

    return pd.Series(total_inconsistency) / pd.Series(total_sum),\
        pd.Series(total_bins_per_pt) / pd.Series(total_sum)
