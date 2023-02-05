from approximation.paa import PAA
from information_embedding_cost.kullback_leibler_divergence import compute_kullback_leibler_divergence
from information_embedding_cost.information_loss import compute_information_loss


"""
This module computes the information embedding cost (IEC) between a time series
and its corresponding symbolic representation based on a SAX variant as
described in the below references [1], [2], [3].


References
----------
[1] Song, W., Wang, Z., Zhang, F., Ye, Y., & Fan, M. (2017). Empirical study of
symbolic aggregate approximation for time series classification. Intelligent
Data Analysis, 21(1), 135-150.

[2] Song, K., Ryu, M., & Lee, K. (2020). Transitional sax representation for
knowledge discovery for time series. Applied Sciences, 10(19), 6980.

[3] Song, W., Wang, Z., Ye, Y., & Fan, M. (2015, August). Empirical studies on
symbolic aggregation approximation under statistical perspectives for knowledge
discovery in time series. In 2015 12th International Conference on Fuzzy
Systems and Knowledge Discovery (FSKD) (pp. 1040-1046). IEEE.
"""


def compute_information_embedding_cost(df_norm, window_size, sax_variant, hist_binning,
                                       df_breakpoints=None, pp_metric="MeanSquaredError",
                                       min_max_scale=True, scale_slopes=True, **symbol_mapping):
    """
    Compute the information embedding cost between time series and their
    corresponding symbolic representations.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series of that the information embedding cost shall be
        computed.
    :param window_size: int
        The size of the window that shall be used to transform the given time
        series into their corresponding PAA representations.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the given time series
        into their corresponding symbolic representations.
    :param hist_binning: HistogramBinning
        The strategy with which the bins for the histograms of the given time
        series shall be built to compute the Kullback-Leibler divergence.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
        Can only be used for the aSAX as given 'sax_variant' and is ignored for
        other SAX variants.
        The individual breakpoints for the PAA representations of the given
        time series dataset that shall be used to transform the respective PAA
        representation into its aSAX representation.
        If None, the respective breakpoints resulting from the k-means
        clustering of the respective PAA points are used.
        This parameter is intended to allow breakpoints based on the
        k-means clustering of the original normalized time series data
        points.
    :param pp_metric: str (default = 'MeanSquaredError')
        The metric that shall be used to measure the information loss. Can be
        any metric of type 'PPMetric' given in 'UpperCamelCase' of type 'str'.
    :param min_max_scale: bool (default = True)
        If true both, the original time series and the reconstructed time
        series are transformed into the range [0, 1] for computing the
        information loss.
        If false, the original time series and the reconstructed time series
        are not transformed into the range [0, 1] on purpose. But, it can be
        that they are still in the range [0, 1] depending on the chosen symbol
        mapping strategy.
    :param scale_slopes: bool (default = True)
        This is only used for the 1d-SAX as given 'sax_variant' for computing
        the information loss and is ignored for other SAX variants.
        It indicates whether to adjust the scaling of the segment slopes. This
        should only be set to 'true' if the symbol mapping strategy for the
        segment means is 'EncodedMinMaxScaling'. Otherwise, the results will be
        distorted.
    :param symbol_mapping: multiple objects of SymbolMapping
        The symbol mapping strategies that shall be used to inverse transform
        the symbolic representation of the given time series dataset for
        computing the information loss.
        The appropriate symbol mapping strategies depend on the given
        'sax_variant'.
    :return: pd.Series of shape (num_ts,)
    """

    df_paa = PAA(window_size).transform(df_norm)
    kl_div = compute_kullback_leibler_divergence(df_paa, df_norm, window_size, sax_variant,
                                                 hist_binning, df_breakpoints)
    info_loss = compute_information_loss(df_paa, df_norm, window_size, sax_variant,
                                         df_breakpoints, pp_metric, min_max_scale,
                                         scale_slopes, **symbol_mapping)
    return kl_div / (1 + info_loss)
