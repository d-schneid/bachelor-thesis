from utils import scale_min_max, get_metric_instance
from discretization.symbol_mapping import EncodedMinMaxScaling
from pp_metrics.pp_metrics import PP_METRICS_MODULE


def _compute_reconstructed_ts(df_paa, df_norm, window_size, sax_variant,
                              df_breakpoints, scale_slopes, **symbol_mapping):
    """
    Compute the reconstructed version of each time series based on one of the
    SAX variants and symbol mapping strategies.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations of the given time series.
    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series for that their reconstructed versions shall be
        computed.
    :param window_size: int
        The size of the window that was used to create the given PAA
        representations.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the given time series
        into their corresponding symbolic representations.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
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
    :param scale_slopes: bool
        This is only used for the 1d-SAX as given 'sax_variant' and is ignored
        for other SAX variants.
        It indicates whether to adjust the scaling of the segment slopes. This
        should only be set to 'true' if the symbol mapping strategy for the
        segment means is 'EncodedMinMaxScaling'. Otherwise, the results will be
        distorted.
    :param symbol_mapping: multiple objects of SymbolMapping
        The symbol mapping strategies that shall be used to inverse transform
        the symbolic representation of the given time series dataset.
        The appropriate symbol mapping strategies depend on the given
        'sax_variant'.
    :return: dataframe of shape (ts_size, num_ts)
    """

    df_reconstructed = sax_variant.transform_inv_transform(df_paa=df_paa, df_norm=df_norm,
                                                           window_size=window_size,
                                                           df_breakpoints=df_breakpoints,
                                                           scale_slopes=scale_slopes, **symbol_mapping)
    return df_reconstructed


def compute_information_loss(df_paa, df_norm, window_size, sax_variant, df_breakpoints=None,
                             pp_metric="MeanSquaredError", min_max_scale=True,
                             scale_slopes=True, **symbol_mapping):
    """
    Compute the information loss between each original time series and its
    corresponding reconstructed version of it based on one of the SAX variants
    and symbol mapping strategies.

    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations of the given time series dataset.
    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series for that the information loss shall be computed.
    :param window_size: int
        The size of the window that shall be used to create the corresponding
        PAA representations of the given time series for computing their
        symbolic representations based on one of the SAX variants.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the given time series
        into their corresponding symbolic representations.
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
        series are transformed into the range [0, 1].
        If false, the original time series and the reconstructed time series
        are not transformed into the range [0, 1] on purpose. But, it can be
        that they are still in the range [0, 1] depending on the chosen symbol
        mapping strategy.
    :param scale_slopes: bool (default = True)
        This is only used for the 1d-SAX as given 'sax_variant' and is ignored
        for other SAX variants.
        It indicates whether to adjust the scaling of the segment slopes. This
        should only be set to 'true' if the symbol mapping strategy for the
        segment means is 'EncodedMinMaxScaling'. Otherwise, the results will be
        distorted.
    :param symbol_mapping: multiple objects of SymbolMapping
        The symbol mapping strategies that shall be used to inverse transform
        the symbolic representation of the given time series dataset.
        The appropriate symbol mapping strategies depend on the given
        'sax_variant'.
    :return: pd.Series of shape (num_ts,)
    """

    if symbol_mapping.get("symbol_mapping_slope", False):
        if isinstance(symbol_mapping["symbol_mapping_slope"], EncodedMinMaxScaling):
            raise ValueError("Do not use the chosen 'symbol_mapping_slope' "
                             "strategy as it leads to meaningless results.")

    df_norm_orig = df_norm
    if min_max_scale:
        df_norm_orig = scale_min_max(df_norm, df_norm.min(), df_norm.max())

    df_reconstructed = _compute_reconstructed_ts(df_paa, df_norm, window_size, sax_variant,
                                                 df_breakpoints, scale_slopes, **symbol_mapping)
    if min_max_scale:
        df_reconstructed = scale_min_max(df_reconstructed, df_reconstructed.min(), df_reconstructed.max())

    pp_metric_instance = get_metric_instance(PP_METRICS_MODULE, pp_metric)

    return pp_metric_instance.compute(df_norm_orig, df_reconstructed)
