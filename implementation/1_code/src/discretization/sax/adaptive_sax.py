import pandas as pd
import numpy as np
import math
import warnings
from sklearn import metrics

from utils import interpolate_segments
from discretization.sax.sax import SAX
from discretization.symbol_mapping import IntervalNormMedian


# the k-means algorithm of the aSAX stops when the relative change of the
# ssq error is below this threshold
THRESHOLD_K_MEANS = 1e-09


def _terminate_k_means(ssq_error, ssq_error_new):
    """
    Determine if the k-means algorithm shall stop based on the SSQ Error.

    :param ssq_error: pd.Series of shape (num_ts,)
        The SSQ Error of the clustering from the previous iteration.
    :param ssq_error_new: pd.Series of shape (num_ts,)
        The SSQ Error of the new clustering from the current iteration.
    :return: bool
        True, if the new SSQ Error is zero for all time series; or the SSQ
        Error does not change anymore compared to the previous SSQ Error for
        all time series; or the relative SSQ Error based on the previous SSQ
        Error is below a threshold for all time series.
        False, otherwise.
    """

    return any([all(ssq_error_new == 0),
                all(ssq_error == ssq_error_new),
                all((abs(ssq_error - ssq_error_new) / ssq_error) < THRESHOLD_K_MEANS)])


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


def _map_intervals(df_paa, df_breakpoints, df_interval_means):
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
    for col_name, col_data in df_paa.items():
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
    Compute the Silhouette coefficient, Calinski-Harabasz index, and
    Davis-Bouldin index for each given PAA representation and its respective
    clustering. These metrics evaluate the internal goodness of the respective
    given clustering.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations whose clusterings shall be evaluated.
    :param df_clustering: dataframe of shape (num_segments, num_ts)
        The individual clusterings of each given PAA representation based on
        the respective interval index (cluster id) of each PAA point.
    :return: np.array of shape (num_ts, 3, 1)
        The values of the three internal evaluation metrics named above for
        each given PAA representation for the given respective clustering.
    """

    num_ts = df_paa.shape[1]
    eval_results = []
    for i in range(num_ts):
        silhouette, calinski_harabasz, davies_bouldin = [], [], []
        paa_values = df_paa.iloc[:, i].to_frame()
        clustering = df_clustering.iloc[:, i]

        silhouette.append(metrics.silhouette_score(paa_values, clustering))
        calinski_harabasz.append(metrics.calinski_harabasz_score(paa_values, clustering))
        davies_bouldin.append(metrics.davies_bouldin_score(paa_values, clustering))

        eval_results.append([silhouette, calinski_harabasz, davies_bouldin])

    return np.array(eval_results)


def eval_k_means(df_paa, min_alphabet_size, max_alphabet_size):
    """
    Evaluate the k-means algorithm of the aSAX for each given PAA
    representation for every alphabet size of the given range. The evaluation
    is run for alphabet sizes from 'min_alphabet_size' (inclusive) to
    'max_alphabet_size' (inclusive).
    This evaluation supports the parameter-free usage of the aSAX
    discretization method by selecting an appropriate alphabet size according
    to the computed evaluation results.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations that shall be (individually) clustered with the
        k-means algorithm of the aSAX.
    :param min_alphabet_size: int
        The lower bound (inclusive) of the aSAX alphabet size for that the
        evaluation shall be run.
    :param max_alphabet_size: int
        The upper bound (inclusive) of the aSAX alphabet size for that the
        evaluation shall be run.
    :return: np.array of shape (num_ts, 4, num_diff_alphabet_sizes)
        There are four internal evaluation metrics that are computed:
        SSQ Error, Silhouette coefficient, Calinski-Harabasz index, and
        Davies-Bouldin index. The returned array contains an array of shape
        (num_diff_alphabet_sizes,) for each of these evaluation metrics within
        each of the given PAA representations and in the listed order above.
    :raises:
        ValueError: If 'min_alphabet_size' > 'max_alphabet_size'.
        ValueError: If 'min_alphabet_size' < 1.
        ValueError: If 'max_alphabet_size' > 26 (number of symbols in Latin
                    alphabet).
    """

    if min_alphabet_size > max_alphabet_size:
        raise ValueError("The maximum alphabet size for evaluation needs to "
                         "be at least as large as the minimum alphabet size")

    num_ts = df_paa.shape[1]
    # 3d-np.array: within each time series 4 internal evaluation metrics that
    # each get computed for every alphabet size of the given range
    eval_results = np.empty([num_ts, 4, 1])
    for k in range(min_alphabet_size, max_alphabet_size + 1):
        a_sax = AdaptiveSAX(alphabet_size=k)
        eval_results_new = a_sax.k_means(df_paa, eval_mode=True)
        # attach evaluation results for current alphabet size to all evaluation
        # results of previous alphabet sizes
        eval_results = np.concatenate([eval_results, eval_results_new], axis=2)

    # delete arbitrary values that were created during initialization by
    # np.empty
    return np.delete(eval_results, 0, axis=2)


class AdaptiveSAX(SAX):
    """
    Adaptive Symbolic Aggreagate Approximation (aSAX).

    :param alphabet_size: int (default = 3)
        The number of symbols in the alphabet that shall be used for
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
        self.name = "aSAX"
        # initialize the k-means algorithm always with the breakpoints used
        # in the classic SAX (equiprobable regions of the standard normal
        # distribution)
        self.init_breakpoints = self.breakpoints

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
        Davies-Bouldin index) of the final clustering (converged) are computed
        for each given PAA representation.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations that shall be discretized based on the
            computed breakpoints, respectively intervals.
            Note: While it is not the standard procedure of the aSAX, the
            original normalized time series data points can be used in the same
            way as the PAA points to compute the respective breakpoints.
        :param eval_mode: bool (default = False)
            The indication if the k-means algorithm shall be run in evaluation
            mode or not.
        :return:
            'eval_mode' = False: dataframe of shape (num_breakpoints, num_ts)
            'eval_mode' = True: np.array of shape (num_ts, 4, 1)
                The values of the four internal evaluation metrics for each
                given PAA representation for the final clustering (converged).
        """

        num_ts = df_paa.shape[1]
        ssq_error, df_breakpoints = self._init_k_means(num_ts)
        ssq_error.index = [column for column in df_paa.columns]
        while True:
            df_interval_means = _compute_interval_means(df_paa, df_breakpoints)
            # new breakpoints
            df_breakpoints = (df_interval_means + df_interval_means.shift(-1)) / 2
            # last row is NaN due to shift
            df_breakpoints.drop(df_breakpoints.tail(1).index, inplace=True)
            # assign each PAA point its interval mean (centroid) and interval
            # index (cluster id)
            df_mapped_interval_means, df_clustering = _map_intervals(
                df_paa, df_breakpoints, df_interval_means)

            ssq_error_new = (df_paa - df_mapped_interval_means).pow(2).sum(axis=0)
            if _terminate_k_means(ssq_error, ssq_error_new):
                if eval_mode:
                    # reshape to 3d-np.array for concatenation
                    ssq_error_new = ssq_error_new.to_numpy().reshape(1, 1, num_ts)
                    # 3d-np.array
                    eval_results = _compute_eval_metrics(df_paa, df_clustering).T
                    return np.concatenate([ssq_error_new, eval_results], axis=1).T
                break
            ssq_error = ssq_error_new

        return df_breakpoints

    def transform(self, df_paa, df_breakpoints=None, *args, **kwargs):
        """
        Transform the PAA representation of each time series into its aSAX
        representation (i.e. assign each PAA representation its respective
        aSAX word) based on individual breakpoints computed by the k-means
        algorithm for each PAA representation (data-adaptive breakpoints) or
        the given breakpoints.
        It does not modify the given PAA representations before computation,
        such as normalization. Therefore, the modification of the PAA
        representations (e.g. normalization) is the responsibility of the user.

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of a time series dataset that shall be
            transformed into their aSAX representations.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            The individual breakpoints for each given PAA representation that
            shall be used to transform the respective PAA representation into
            its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            This parameter is intended to allow breakpoints based on the
            k-means clustering of the original normalized time series data
            points.
        :return:
            dataframe of shape (num_segments, num_ts)
            dataframe of shape (num_breakpoints, num_ts)
        """

        df_breakpoints = self.k_means(df_paa) if df_breakpoints is None else df_breakpoints
        a_sax_reprs = []
        for i in range(df_paa.shape[1]):
            self.breakpoints = np.array(df_breakpoints.iloc[:, i])
            # transform time series by time series, because of individual
            # breakpoints
            df_sax = super().transform(df_paa.iloc[:, i].to_frame())
            a_sax_reprs.append(df_sax)
        return pd.concat(a_sax_reprs, axis=1), df_breakpoints

    def inv_transform(self, df_a_sax, ts_size, window_size, symbol_mapping,
                      df_breakpoints=None, *args, **kwargs):
        """
        Approximate the original time series dataset by transforming its aSAX
        representations into a time series dataset with the same size by
        assigning each point the symbol value of its segment based on the
        individual breakpoints of each aSAX representation.

        :param df_a_sax: dataframe of shape (num_segments, num_ts)
            The aSAX representation of the time series dataset.
        :param ts_size: int
            The size of the original time series.
        :param window_size: int
            The size of the segments with which the given aSAX representations
            were created.
        :param symbol_mapping: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the aSAX symbols.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            The individual breakpoints for each aSAX representation that were
            used to create it.
        :return:
            dataframe of shape (ts_size, num_ts)
        """

        if isinstance(symbol_mapping, IntervalNormMedian):
            warnings.warn("Use the chosen 'symbol_mapping' strategy with "
                          "caution, because in the aSAX, the breakpoint "
                          "intervals are not (primarily) determined based on "
                          "a Gaussian distribution.")

        # use breakpoints of classic SAX (equiprobable regions of the standard
        # normal distribution) if no breakpoints are given
        self.breakpoints = self.init_breakpoints
        inv_a_sax_reprs = []
        for idx in range(df_a_sax.shape[1]):
            if df_breakpoints is not None:
                self.breakpoints = np.array(df_breakpoints.iloc[:, idx])
            # inverse transform time series by time series with the individual
            # breakpoints that were used for the transformation into the given
            # aSAX representation
            df_inv_a_sax = super().inv_transform(df_a_sax.iloc[:, idx].to_frame(),
                                                 ts_size, window_size,
                                                 symbol_mapping, idx)
            inv_a_sax_reprs.append(df_inv_a_sax)

        return pd.concat(inv_a_sax_reprs, axis=1)

    def transform_inv_transform(self, df_paa, df_norm, window_size, df_breakpoints=None, **symbol_mapping):
        df_a_sax, df_breakpoints = self.transform(df_paa, df_breakpoints)
        return self.inv_transform(df_a_sax, df_norm.shape[0], window_size,
                                  **symbol_mapping, df_breakpoints=df_breakpoints)

    def transform_to_symbolic_ts(self, df_paa, df_norm, window_size, df_breakpoints=None):
        df_a_sax, df_breakpoints = self.transform(df_paa, df_breakpoints)
        return interpolate_segments(df_a_sax, df_norm.shape[0], window_size)

    def transform_to_symbolic_repr_only(self, df_paa, df_norm, window_size, df_breakpoints):
        df_a_sax, df_breakpoints = self.transform(df_paa=df_paa, df_norm=df_norm,
                                                  window_size=window_size,
                                                  df_breakpoints=df_breakpoints)
        return df_a_sax

    def compute_breakpoints(self, df_norm):
        """
        Compute individual breakpoints for each time series based on the
        k-means algorithm on the normalized time series points.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series for that individual breakpoints shall be computed.
        :return: dataframe of shape (num_breakpoints, num_ts)
        """

        return self.k_means(df_norm)
