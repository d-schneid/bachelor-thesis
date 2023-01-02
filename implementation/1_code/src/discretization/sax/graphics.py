import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import qqplot

from utils.utils import constant_segmentation, interpolate_segments


def plot_norm_qq(df_orig, df_norm, num_column, scale=1):
    """
    PLot the original time series and its z-normalized version along with their
    corresponding Q-Q plots by comparing it to a Gaussian distribution of
    N(0, 'scale').

    :param df_orig: dataframe of shape (ts_size, num_ts)
        The original time series dataset.
    :param df_norm: dataframe of shape (ts_size, num_ts)
        The z-normalized version of the given original time series dataset.
    :param num_column: int
        The number of the column (i.e. the time series) that shall be plotted.
    :param scale: float (default = 1)
        The variance of the Gaussian distribution that is used for comparison
        for building the Q-Q plots.
        Usually, this value only deviates from the set default value for the
        quantization of the slope values in the 1d-SAX discretization method.
    :return:
        None
    """

    # original time series
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_orig.index, df_orig.iloc[:, num_column])
    ax.set_title(f'Original Time Series - Column {df_orig.columns[num_column]}')
    ax.set_xlabel('Time')
    plt.show()

    # z-normalized version of original time series
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_norm.index, df_norm.iloc[:, num_column])
    ax.set_title(f'Z-Normalized Time Series - Column {df_norm.columns[num_column]}')
    ax.set_xlabel('Time')
    plt.show()

    # Q-Q plot of original time series
    fig = plt.figure(figsize=(13, 5))
    position = 121
    ax = fig.add_subplot(position)
    qqplot(df_orig.iloc[:, num_column], scale=scale, line='45', ax=ax,
           c='royalblue', markersize=0.5)
    ax.set_xlabel('Gaussian Theoretical Quantiles')
    ax.set_ylabel('Time Series Quantiles')
    ax.set_title(f'Q-Q plot - Original Time Series\nColumn {df_orig.columns[num_column]}')

    # Q-Q plot of z-normalized version of original time series
    position = 122
    ax = fig.add_subplot(position)
    qqplot(df_norm.iloc[:, num_column], scale=scale, line='45', ax=ax,
           c='royalblue', markersize=0.5)
    ax.set_xlabel('Gaussian Theoretical Quantiles')
    ax.set_ylabel('Time Series Quantiles')
    ax.set_title(f'Q-Q plot - Z-Normalized Time Series\nColumn {df_norm.columns[num_column]}')

    plt.show()


# TODO: comment? not usable for 1d-SAX, because segments are linear functions and not constant functions
def plot_paa_sax_symbols(df_norm, df_paa, df_sax, breakpoints, alphabet_size,
                         window_size, num_column):
    """
    Plot a PAA representation and its normalized time series along with the
    respective symbol for each segment based on the given breakpoints and its
    SAX representation.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset of which one PAA representation
        shall be plotted.
    :param df_paa: dataframe of shape (num_segments, num_ts)
        The PAA representations of which one shall be plotted.
    :param df_sax: dataframe of shape (num_segments, num_ts)
        The SAX representations of the given PAA representations.
    :param breakpoints: np.array
        The breakpoints on which the given SAX representations are based.
    :param alphabet_size: int
        The size of the alphabet on which the given SAX representations are based.
    :param window_size: int
        The size of the window on which the given PAA representations are based.
    :param num_column: int
        The number of the column (i.e. the time series and its PAA
        representation) that shall be plotted.
    :return:
        None
    """

    ts_size = df_norm.shape[0]
    start, end, num_segments = constant_segmentation(ts_size, window_size)
    t_segment_middle = (start + end - 1) / 2

    df_paa_interpolated = interpolate_segments(df_paa, ts_size, window_size)

    fig, ax = plt.subplots(figsize=(13, 5))

    # plot breakpoints as horizontal lines
    for line in breakpoints:
        ax.axhline(line, c='crimson', alpha=0.4, ls='--', lw=0.7)

    # plot respective symbol of each segment
    segment_voffset = 0.1
    for t, symbol in enumerate(df_sax.iloc[:, num_column]):
        ax.text(t_segment_middle[t], df_paa.iloc[t, num_column] + segment_voffset,
                f'{symbol}', c='crimson', ha='center', fontsize=15)

    # plot PAA
    ax.step(df_paa_interpolated.index, df_paa_interpolated.iloc[:, num_column],
            where='post', c='royalblue', alpha=0.9)
    ax.plot([df_paa_interpolated.index[-1], df_norm.index[-1]],
            [df_paa_interpolated.iloc[-1, num_column],
             df_paa_interpolated.iloc[-1, num_column]], c='royalblue', alpha=0.9)

    # plot normalized time series
    ax.plot(df_norm.index, df_norm.iloc[:, num_column], alpha=0.5)

    # plot parameters
    ax.set_title(f'Representation based on window size = {window_size} & '
                 f'alphabet size = {alphabet_size}\nColumn {df_norm.columns[num_column]}')
    ax.set_xlabel('Time')
    ax.set_xlim((df_norm.index[0], df_norm.index[-1]))

    plt.show()


def plot_sax_symbols(df_norm, df_sax, alphabet_avg, window_size, num_column,
                     alphabet_slope=None):
    """
    Plot SAX symbols as a step function based on the corresponding segments.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset from which the given SAX
        representations were created.
    :param df_sax: dataframe of shape (num_segments, num_ts)
        The SAX representations of which one shall be plotted as step function.
    :param alphabet_avg: np.array
        The alphabet with which the segment means were quantized for creating
        the given SAX representations.
    :param window_size: int
        The size of the window with which the PAA representations for the given
        SAX representations were computed.
    :param num_column: int
        The number of the column (i.e. the SAX representation) that shall be
        plotted as a step function.
    :param alphabet_slope: np.array
        The alphabet with which the segment slopes were quantized for creating
        the given 1d-SAX representations.
    :return:
        None
    """

    alphabet = alphabet_avg
    if alphabet_slope is not None:
        alphabet = np.array([symbol_avg + symbol_slope
                             for symbol_avg in alphabet_avg
                             for symbol_slope in alphabet_slope])

    ts_size = df_norm.shape[0]
    df_sax_interpolated = interpolate_segments(df_sax, ts_size, window_size)

    fig, ax = plt.subplots(figsize=(13, 5))

    # assign a value to each (composed) symbol for plotting purposes
    d = dict(zip([np.where(alphabet == symbol)[0][0] for symbol in alphabet], alphabet))
    reverse_d = {v: k for k, v in d.items()}
    symbol_values = [reverse_d[symbol] for symbol in df_sax_interpolated.iloc[:, num_column]]

    # plot step function based on symbols (values)
    ax.step(df_sax_interpolated.index, symbol_values, where='post', c='k',
            linewidth=2, alpha=0.8)
    # plot final segment
    ax.plot([df_sax_interpolated.index[-1], df_norm.index[-1]],
            [symbol_values[-1], symbol_values[-1]], c='k', linewidth=2, alpha=0.8)

    # plot parameters
    ax.set_title(f'SAX Representation based on window size = {window_size} & '
                 f'alphabet size = {alphabet.size}\nColumn {df_sax_interpolated.columns[num_column]}')

    plt.yticks(range(len(alphabet)), alphabet)
    plt.grid()
    plt.show()


def plot_sax_variants(df_norm, df_paa_inv, df_sax_inv, df_a_sax_inv,
                      df_one_d_sax_inv, window_size, sax_alphabet_size,
                      a_sax_alphabet_size, one_d_sax_alphabet_size_avg,
                      one_d_sax_alphabet_size_slope, num_column):
    """
    Plot an original normalized time series, its PAA representation, its SAX
    representation, and its 1d-SAX representation.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset from which one time series shall be
        plotted.
    :param df_paa_inv: dataframe of shape (ts_size, num_ts)
        The inverse PAA representations of the given normalized time series
        dataset.
    :param df_sax_inv: dataframe of shape (ts_size, num_ts)
        The inverse SAX representations of the given normalized time series
        dataset.
    :param df_a_sax_inv: dataframe of shape (ts_size, num_ts)
        The inverse Adaptive SAX representations of the given normalized time
        series dataset.
    :param df_one_d_sax_inv: dataframe of shape (ts_size, num_ts)
        The inverse 1d-SAX representations of the given normalized time series
        dataset.
    :param window_size: int
        The size of the window that was used to create the PAA representations
        of the given normalized time series dataset.
    :param sax_alphabet_size: int
        The size of the alphabet that was used to create the SAX
        representations corresponding to the given normalized time series
        dataset.
    :param a_sax_alphabet_size: int
        The size of the alphabet that was used to create the Adaptive SAX
        representations corresponding to the given normalized time series
        dataset.
    :param one_d_sax_alphabet_size_avg: int
        The size of the alphabet for the segment means that was used to create
        the 1d-SAX representations corresponding to the given normalized time
        series dataset.
    :param one_d_sax_alphabet_size_slope: int
        The size of the alphabet for the segment slopes that was used to create
        the 1d-SAX representations corresponding to the given normalized time
        series dataset.
    :param num_column: int
        The number of the column (i.e. the time series and its representations)
        that shall be plotted.
    :return:
        None
    """

    # plot raw time series
    plt.figure()
    position = 231
    plt.subplot(position)
    plt.plot(df_norm.iloc[:, num_column], "b-")
    plt.title("Raw time series")

    # plot SAX
    position = 232
    plt.subplot(position)
    plt.plot(df_norm.iloc[:, num_column], "b-", alpha=0.4)
    plt.plot(df_sax_inv.iloc[:, num_column], "b-")
    plt.title(f"SAX, {sax_alphabet_size} symbols")

    # plot Adaptive SAX
    position = 233
    plt.subplot(position)
    plt.plot(df_norm.iloc[:, num_column], "b-", alpha=0.4)
    plt.plot(df_a_sax_inv.iloc[:, num_column], "b-")
    plt.title(f"Adaptive SAX, {a_sax_alphabet_size} symbols")

    # plot PAA
    position = 234
    plt.subplot(position)
    plt.plot(df_norm.iloc[:, num_column], "b-", alpha=0.4)
    plt.plot(df_paa_inv.iloc[:, num_column], "b-")
    plt.title(f"PAA, window size {window_size}")

    # plot 1d-SAX
    position = 235
    plt.subplot(position)
    plt.plot(df_norm.iloc[:, num_column], "b-", alpha=0.4)
    plt.plot(df_one_d_sax_inv.iloc[:, num_column], "b-")
    plt.title("1d-SAX, "
              f"{one_d_sax_alphabet_size_avg * one_d_sax_alphabet_size_slope} "
              "symbols "
              f"({one_d_sax_alphabet_size_avg}x{one_d_sax_alphabet_size_slope})")

    plt.tight_layout()
    plt.show()


def plot_eval_k_means(ssq_error, silhouette, calinski_harabasz, davies_bouldin,
                      min_alphabet_size, max_alphabet_size):
    """
    Plot the given internal evaluation metrics of the k-means clustering used
    in the aSAX discretization method.

    :param ssq_error: list of len(ssq_error) = 'max_alphabet_size' - ('min_alphabet_size' - 1)
        Contains the values of the respective SSQ error for each alphabet size.
    :param silhouette: list of len(ssq_error) = 'max_alphabet_size' - ('min_alphabet_size' - 1)
        Contains the values of the respective Silhouette coefficient for each alphabet size.
    :param calinski_harabasz: list of len(ssq_error) = 'max_alphabet_size' - ('min_alphabet_size' - 1)
        Contains the values of the respective Calinski-Harabasz index for each alphabet size.
    :param davies_bouldin: list of len(ssq_error) = 'max_alphabet_size' - ('min_alphabet_size' - 1)
        Contains the values of the respective Davies-Bouldin index for each alphabet size.
    :param min_alphabet_size: float
        The lower bound (inclusive) of the aSAX alphabet size for that the
        evaluation was run.
    :param max_alphabet_size: float
        The upper bound (inclusive) of the aSAX alphabet size for that the
        evaluation was run.
    :return:
        None
    """

    x_ticks = range(min_alphabet_size, max_alphabet_size + 1)
    x_label = "alphabet size"

    plt.figure()
    position = 221
    plt.subplot(position)
    plt.plot(x_ticks, ssq_error, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("SSQ-Error")

    position = 222
    plt.subplot(position)
    plt.plot(x_ticks, silhouette, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Silhouette")

    position = 223
    plt.subplot(position)
    plt.plot(x_ticks, calinski_harabasz, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Calinski-Harabasz")

    position = 224
    plt.subplot(position)
    plt.plot(x_ticks, davies_bouldin, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Davies-Bouldin")

    plt.tight_layout()
    plt.show()
