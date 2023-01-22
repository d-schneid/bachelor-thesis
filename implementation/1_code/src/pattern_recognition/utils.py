from approximation.paa import PAA
from discretization.sax.abstract_sax import linearize_sax_word
from pattern_recognition.motif_discovery.utils import _encode_symbols


def get_linearized_encoded_sax(df_norm, window_size, sax_variant):
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
    :return:
        df_sax_linearized_encoded: dataframe of shape
            (num_segments * symbols_per_segment, num_ts)
            The linearized and encoded SAX representation of the given time
            series dataset.
        df_sax: dataframe of shape (num_segments, num_ts)
            The SAX representation of the given time series dataset.
    """

    paa = PAA(window_size=window_size)
    df_paa = paa.transform(df_norm)

    df_sax = sax_variant.transform(df_paa=df_paa, df_norm=df_norm, window_size=window_size)
    # eSAX and aSAX additionally return data for inverse transformation
    # extract only symbolic transformation
    if type(df_sax) is tuple:
        df_sax = df_sax[0]
    df_sax_linearized = linearize_sax_word(df_sax, sax_variant.symbols_per_segment)
    df_sax_linearized_encoded = _encode_symbols([[df_sax_linearized]], sax_variant)[0][0]

    return df_sax_linearized_encoded, df_sax
