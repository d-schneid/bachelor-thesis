from abc import ABC, abstractmethod


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
        Compute the inconsistency rate of the current time series point.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the current time
            series point with their corresponding bins.
        :return: float
        """

        inconsistency = epsilon_pts_bins.size - epsilon_pts_bins.value_counts().max()
        return inconsistency / epsilon_pts_bins.size

    def compute_sum(self, epsilon_pts_bins):
        """
        Helps to count the number of points in the time series.

        :param epsilon_pts_bins: pd.Series
            Is ignored, just here for API consistency.
            Contains all points in the epsilon neighborhood of the current time
            series point with their corresponding bins.
        :return: int
        """

        return 1

    def compute_num_bins(self, epsilon_pts_bins):
        """
        Compute the number of different discretization bins the current time
        series point was assigned to.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the current time
            series point with their corresponding bins.
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
        Compute the absolute inconsistency value of the current time series
        point.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the current time
            series point with their corresponding bins.
        :return: int
        """

        inconsistency = epsilon_pts_bins.size - epsilon_pts_bins.value_counts().max()
        return inconsistency

    def compute_sum(self, epsilon_pts_bins):
        """
        Helps to sum up the weights of the time series points.
        The weight of a time series point is the number of its occurrences in
        the time series.

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the current time
            series point with their corresponding bins.
        :return: int
        """

        return epsilon_pts_bins.size

    def compute_num_bins(self, epsilon_pts_bins):
        """
        Compute the number of different discretization bins the current time
        series point was assigned to; weighted by the occurrences of the
        respective time series point in the time series

        :param epsilon_pts_bins: pd.Series
            Contains all points in the epsilon neighborhood of the current time
            series point with their corresponding bins.
        :return: int
        """

        return epsilon_pts_bins.size * len(epsilon_pts_bins.value_counts())
