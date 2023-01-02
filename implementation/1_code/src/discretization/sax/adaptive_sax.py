import pandas as pd
import numpy as np
import math
from sklearn import metrics

from discretization.sax.sax import SAX
from discretization.sax.graphics import plot_eval_k_means


# the k-means algorithm of the aSAX stops when the relative change of the
# ssq error is below this threshold
THRESHOLD_K_MEANS = 1e-09


def _find_min_above_threshold(paa_values, threshold):
    """
    Find minimum of values above a given threshold.

    :param paa_values: dataframe of shape (num_segments, num_ts)
        The PAA values for which the minimum above the given
        threshold shall be found.
    :param threshold: float
        The threshold that shall be used as a (strict) lower bound to search
        for the minimum value.
    :return: float
        If there does not exist any value above the given threshold, it returns
        'inf'.
    """

    filtered_paa_values = [x for x in paa_values if x >= threshold]
    if not filtered_paa_values:
        return float("inf")
    return min(filtered_paa_values)


def _map_interval_means(df_paa, df_breakpoints, df_interval_means):
    """
    Assign each PAA point its respective interval mean, respectively centroid
    and its respective interval index, respectively cluster id.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations that get assigned their respective interval
        mean.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
        The respective breakpoints of the interval for each time series.
    :param df_interval_means: dataframe of shape (num_intervals, num_ts)
        The interval means based on the given PAA values for each time series.
        The number of intervals corresponds to the alphabet size.
    :return:
        mapped_interval_means: dataframe of shape (num_segments, num_ts)
        clustering: dataframe of shape (num_segments, num_ts)
    """

    mapped_interval_means, clustering = [], []
    for i in range(df_paa.shape[1]):
        # assign each PAA point its respective interval index
        breakpoint_idx = pd.Series(
            np.searchsorted(df_breakpoints.iloc[:, i], df_paa.iloc[:, i],
                            side="right"), index=df_paa.index)
        clustering.append(breakpoint_idx)

        mapping = df_interval_means.iloc[:, i].to_dict()
        # assign each PAA point its respective interval mean (centroid)
        breakpoint_idx.replace(to_replace=mapping, inplace=True)
        mapped_interval_means.append(breakpoint_idx)

    mapped_interval_means = pd.concat(mapped_interval_means, axis=1, keys=df_paa.columns)
    clustering = pd.concat(clustering, axis=1, keys=df_paa.columns)
    return mapped_interval_means, clustering


def _check_interval_means(interval_point_means, paa_values, breakpoints):
    """
    In the k-means algorithm, interval means (centroids) could not be defined
    if there are no PAA values in the respective interval. Therefore, some
    proxy PAA value is assigned for missing interval means to make the k-means
    algorithm more robust. This proxy PAA value is either the minimum value
    above the empty interval or the maximum value below the empty interval.
    The idea is that such points do not tend to belong to their current
    cluster, because they are located at the edge of it. Therefore, if they are
    selected as new centroids, they make up a new cluster splitting up their
    current cluster which results in two more dense clusters than their current
    cluster before was.
    The invariant that is maintained is that the proxy PAA value for a missing
    interval is always at least as large as the proxy PAA values for missing
    intervals previously encountered in the current iteration.
    Caveat: This is only a heuristic and not meant to be the optimal solution.
    It is chosen for the sole purpose to keep the k-means algorithm running in
    case of empty intervals.
    Further, especially for realistic real-life and long time series, it is
    unlikely that empty intervals will occur. Also, due to z-normalization and
    the initial breakpoints of the k-means algorithm (the one of the classic
    SAX).

    :param interval_point_means: pd.Series of shape (<= num_intervals,)
        The found interval means given the PAA values.
    :param paa_values: pd.Series of shape (num_segments,)
        The PAA values from which the interval means are computed.
    :param breakpoints: pd.Series of shape (num_breakpoints,)
        The breakpoints that define the interval boundaries.
    :return: pd.Series of shape (num_intervals,)
        It always returns a respective interval mean for each existing
        interval. The number of intervals corresponds to the alphabet size.
    """

    alphabet_size = len(breakpoints) + 1
    # correct index numbers if there would be no empty intervals
    compare_idx = pd.Index(range(alphabet_size))
    missing_intervals_idx = list(compare_idx.difference(interval_point_means.index))
    missing_interval_means = []
    # there are empty intervals
    if missing_intervals_idx:
        for idx in missing_intervals_idx:
            # last interval is empty, there can be only PAA values below
            # max PAA value is the closest PAA value from the last interval
            if idx == breakpoints.size:
                proxy_interval_mean = max(paa_values)
            # search for closest PAA value above or below of missing interval
            else:
                # minimum PAA value above the missing interval
                proxy_interval_mean = _find_min_above_threshold(
                    paa_values, breakpoints.iloc[idx])
                if math.isinf(proxy_interval_mean):
                    # maximum PAA value below the missing interval
                    proxy_interval_mean = -_find_min_above_threshold(
                        -1 * paa_values, -breakpoints[idx-1])

            # at least one of min above and max below exist,
            # because otherwise there would be no PAA values
            missing_interval_means.append(proxy_interval_mean)

        missing_interval_means = pd.Series(missing_interval_means,
                                           index=missing_intervals_idx)
        return pd.concat([interval_point_means, missing_interval_means]).sort_index()

    # there are no empty intervals
    return interval_point_means


def _compute_interval_means(df_paa, df_breakpoints):
    """
    Compute the interval means of the respective PAA values in each interval
    for each time series separately. The intervals correspond to the clusters
    and the interval means to the centroids of the clusters in the k-means
    terminology.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA values of time series that are used to compute the interval
        means.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
        The respective breakpoints of the intervals for each time series.
    :return: dataframe of shape (num_intervals, num_ts)
        The number of intervals corresponds to the alphabet size.
    """

    interval_means = []
    idx = 0
    for col_num, col_data in df_paa.items():
        # assign each PAA point its respective interval index
        breakpoint_idx = np.searchsorted(df_breakpoints.iloc[:, idx], col_data,
                                         side="right")
        interval_point_means = col_data.groupby(by=breakpoint_idx).mean()
        interval_means.append(_check_interval_means(
            interval_point_means, col_data, df_breakpoints.iloc[:, idx]))
        idx += 1
    return pd.concat(interval_means, axis=1)


def _compute_eval_metrics(df_paa, df_clustering):
    """
    Compute the average Silhouette coefficient, Calinski-Harabasz index, and
    Davis-Bouldin index across all given PAA representations and their
    respective clustering. These metrics evaluate the internal goodness of the
    given clustering.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations whose clustering shall be evaluated.
    :param df_clustering: dataframe of shape (num_segments, num_ts)
        The individual clustering of each given PAA representation based on the
        respective interval index (cluster id) of each PAA point.
    :return:
        float, float, float
    """

    num_ts = df_paa.shape[1]
    silhouette, calinski_harabasz, davies_bouldin = 0, 0, 0
    for i in range(num_ts):
        paa_values = df_paa.iloc[:, i].to_frame()
        clustering = df_clustering.iloc[:, i]
        silhouette += metrics.silhouette_score(paa_values, clustering)
        calinski_harabasz += metrics.calinski_harabasz_score(paa_values, clustering)
        davies_bouldin += metrics.davies_bouldin_score(paa_values, clustering)

    return silhouette/num_ts, calinski_harabasz/num_ts, davies_bouldin/num_ts


def eval_k_means(df_paa, min_alphabet_size, max_alphabet_size):
    """
    Evaluate the k-means algorithm of the aSAX for different sizes of its
    alphabet. The evaluation is run for alphabet sizes from 'min_alphabet_size'
    (inclusive) to 'max_alphabet_size' (inclusive).
    The evaluation results will be plotted for visual inspection. This supports
    the parameter-free usage of the aSAX discretization method by selecting an
    appropriate alphabet size according to the plotted evaluation results.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations that shall be (individually) clustered with the
        k-means algorithm of the aSAX.
    :param min_alphabet_size: int
        The lower bound (inclusive) of the aSAX alphabet size for that the
        evaluation shall be run.
    :param max_alphabet_size: int
        The upper bound (inclusive) of the aSAX alphabet size for that the
        evaluation shall be run.
    :return:
        None
    :raises:
        ValueError: If 'min_alphabet_size' > 'max_alphabet_size'.
        ValueError: If 'min_alphabet_size' < 1.
        ValueError: If 'max_alphabet_size' > 26 (number of letters in Latin
                    alphabet).
    """

    if min_alphabet_size > max_alphabet_size:
        raise ValueError("The maximum alphabet size for evaluation needs to "
                         "be at least as large as the minimum alphabet size")

    ssq_error, silhouette, calinski_harabasz, davies_bouldin = [], [], [], []
    for k in range(min_alphabet_size, max_alphabet_size + 1):
        a_sax = AdaptiveSAX(alphabet_size=k)
        ssq_err, silh, c_h, d_b = a_sax.k_means(df_paa, eval_mode=True)
        ssq_error.append(ssq_err)
        silhouette.append(silh)
        calinski_harabasz.append(c_h)
        davies_bouldin.append(d_b)
    plot_eval_k_means(ssq_error, silhouette, calinski_harabasz,
                      davies_bouldin, min_alphabet_size, max_alphabet_size)


class AdaptiveSAX(SAX):
    """
    Adaptive Symbolic Aggreagate Approximation (aSAX).

    :param alphabet_size: int (default = 3)
        The number of letters in the alphabet that shall be used for
        discretization. The alphabet starts from 'a' and ends with 'z' at the
        latest.
    :raises:
        ValueError: If the size of the alphabet is above 26 or below 1.

    References
    ----------
    [1] Pham, Ninh D., Quang Loc Le, and Tran Khanh Dang. "Two novel adaptive
    symbolic representations for similarity search in time series databases."
    2010 12th International Asia-Pacific Web Conference. IEEE, 2010.
    """

    def __init__(self, alphabet_size=3):
        super().__init__(alphabet_size=alphabet_size)
        # initialize the k-means algorithm always with the breakpoints used
        # in the classic SAX (equiprobable regions of the standard normal
        # distribution)
        self.init_breakpoints = self.breakpoints_avg

    def _init_k_means(self, num_ts):
        """
        Initialize k-means algorithm for each time series. The error is
        initialized with 'infinity' and the breakpoints are initialized with
        the breakpoints of the classic SAX.

        :param num_ts: int
            The number of time series on which the k-means algorithm shall be
            applied.
        :return:
            ssq_error: pd.Series of shape (num_ts,)
            df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
        """

        ssq_error = pd.Series([float("inf") for _ in range(num_ts)])
        df_breakpoints = pd.DataFrame([self.init_breakpoints for _ in range(num_ts)]).T
        return ssq_error, df_breakpoints

    def k_means(self, df_paa, eval_mode=False):
        """
        Compute the breakpoints of the respective intervals for the SAX
        discretization of each given PAA representation with the k-means
        algorithm based on the given PAA values. The intervals correspond to
        the clusters and the interval means to the centroids of the clusters in
        the k-means terminology.
        Can be run in evaluation mode ('eval_mode') where internal evaluation
        metrics (SSQ Error, Silhouette coefficient, Calinski-Harabasz index,
        Davies-Bouldin index) of the final clustering are computed averaged
        across the given PAA representations.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations that shall be discretized based on the
            computed breakpoints, respectively intervals.
        :param eval_mode: bool (default = False)
            The indication if the k-means algorithm shall be run in evaluation
            mode or not.
        :return:
            'eval_mode' = False: dataframe of shape (num_breakpoints, num_ts)
            'eval_mode' = True: float, float, float, float
        """

        ssq_error, df_breakpoints = self._init_k_means(df_paa.shape[1])
        ssq_error.index = [column for column in df_paa.columns]
        while True:
            df_interval_means = _compute_interval_means(df_paa, df_breakpoints)
            # new breakpoints
            df_breakpoints = (df_interval_means + df_interval_means.shift(-1)) / 2
            # last row is NaN due to shift
            df_breakpoints.drop(df_breakpoints.tail(1).index, inplace=True)
            # assign each PAA point its interval mean (centroid) and interval
            # index (cluster id)
            df_mapped_interval_means, df_clustering = _map_interval_means(
                df_paa, df_breakpoints, df_interval_means)

            ssq_error_new = (df_paa - df_mapped_interval_means).pow(2).sum(axis=0)
            if all((abs(ssq_error - ssq_error_new) / ssq_error) < THRESHOLD_K_MEANS):
                if eval_mode:
                    silhouette, calinski_harabasz, davies_bouldin =\
                        _compute_eval_metrics(df_paa, df_clustering)
                    return ssq_error_new.mean(), silhouette,\
                        calinski_harabasz, davies_bouldin
                break
            ssq_error = ssq_error_new

        return df_breakpoints

    def transform(self, df_paa):
        """
        Transform the PAA representation of each time series into its aSAX
        representation (i.e. assign each PAA representation its respective
        aSAX word) based on individual breakpoints computed by the k-means
        algorithm for each PAA representation (data-adaptive breakpoints).
        It does not modify the given PAA representations before computation,
        such as normalization. Therefore, the modification of the PAA
        representations (e.g. normalization) is the responsibility of the user.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of a time series dataset that shall be
            transformed into their aSAX representations.
        :return:
            dataframe of shape (num_segments, num_ts)
        """

        df_breakpoints = self.k_means(df_paa)
        a_sax_reprs = []
        for i in range(df_paa.shape[1]):
            self.breakpoints_avg = np.array(df_breakpoints.iloc[:, i])
            # transform column by column, because of individual breakpoints
            df_sax = super().transform(df_paa.iloc[:, i].to_frame())
            a_sax_reprs.append(df_sax)
        return pd.concat(a_sax_reprs, axis=1)