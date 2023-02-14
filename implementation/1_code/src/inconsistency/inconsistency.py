import pandas as pd

from approximation.paa import PAA


def compute_inconsistency_metrics(df_norm, window_size, sax_variant, inconsistency_mode,
                                  epsilon, df_breakpoints=None):
    """
    Compute the unweighted or weighted mean inconsistency rate per time series
    point and the unweighted or weighted mean number of discretization bins per
    time series point.
    The inconsistency metrics are computed by iterating and evaluating each
    point of a time series based on its inconsistency.

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
    :param epsilon: float
        The tolerance for time series points to be considered equal.
        All points that are in the neighborhood of the current point:
        current_point - epsilon <= current_point <= current_point + epsilon
        are considered equal to this point.
        This is needed, because in most time series floating points are used.
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
        for pt_idx in range(len(current_ts)):
            current_pt = current_ts[pt_idx]
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
