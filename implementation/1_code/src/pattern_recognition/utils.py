from approximation.paa import PAA
from discretization.sax.abstract_sax import linearize_sax_word
from pattern_recognition.motif_discovery.utils import _encode_symbols


def get_linearized_encoded_sax(df_norm, window_size, sax_variant, df_breakpoints=None):
    """
    Transform the given time series dataset into its linearized and encoded SAX
    representation based on the given SAX variant.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series dataset that shall be transformed into its linearized
        and encoded SAX representation.
    :param window_size: int
        The size of the window that shall be used to transform the given time
        series dataset into its PAA representation.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the PAA representation
        of the given time series dataset into its SAX representation.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
            Is ignored for all SAX variants except the aSAX.
            The individual breakpoints for the given PAA representations of the
            given time series dataset that shall be used to transform the
            respective PAA representation into its aSAX representation.
            If None, the respective breakpoints resulting from the k-means
            clustering of the respective PAA points are used.
            With this parameter breakpoints based on the k-means clustering of
            the original normalized time series data points are also possible.
    :return:
        df_sax_linearized_encoded: dataframe of shape
            (num_segments * symbols_per_segment, num_ts)
            The linearized and encoded symbolic representation of the given
            time series dataset.
        df_sax: dataframe of shape (num_segments, num_ts)
            The symbolic representation of the given time series dataset.
    """

    paa = PAA(window_size=window_size)
    df_paa = paa.transform(df_norm)

    df_sax = sax_variant.transform_to_symbolic_repr_only(df_paa=df_paa, df_norm=df_norm,
                                                         window_size=window_size, df_breakpoints=df_breakpoints)
    df_sax_linearized = linearize_sax_word(df_sax, sax_variant.symbols_per_segment)
    df_sax_linearized_encoded = _encode_symbols([[df_sax_linearized]], sax_variant)[0][0]

    return df_sax_linearized_encoded, df_sax
