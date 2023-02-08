import numpy as np
import matplotlib.pyplot as plt


def plot_inconsistency_metrics(inconsistency, num_bins, x_ticks, num_col, x_label, x_scale):
    """
    Plot the inconsistency metrics for a time series computed on different
    values for a parameter (e.g. alphabet size, epsilon).

    :param inconsistency: dataframe of shape (len(x_ticks), num_ts)
        The mean inconsistency rate for each time series in each round of a
        different value for a parameter (e.g. alphabet size, epsilon).
    :param num_bins: dataframe of shape (len(x_ticks) , num_ts)
        The mean number of discretization bins for each time series in each
        round of a different value for a parameter (e.g. alphabet size,
        epsilon).
    :param x_ticks: np.array
        Contains the different values of a parameter (e.g. alphabet size,
        epsilon) the inconsistency metrics are computed on in ascending order.
    :param num_col: int
        Indicates the time series for which the corresponding inconsistency
        metrics shall be plotted.
    :param x_label: str
        The label of the x-axis.
    :param x_scale: str
        The scale that shall be used for the x-axis (e.g. linear, log).
    :return: None
    """

    inconsistency = inconsistency.iloc[:, num_col]
    num_bins = num_bins.iloc[:, num_col]
    plt.figure(figsize=(15, 7))

    position = 211
    plt.subplot(position)
    plt.plot(x_ticks, inconsistency, "b-")
    plt.xscale(x_scale)
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Mean Inconsistency Rate")

    position = 212
    plt.subplot(position)
    plt.plot(x_ticks, num_bins, "b-")
    plt.xscale(x_scale)
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Mean Number of Discretization Bins")

    plt.tight_layout()
    plt.show()


def plot_comparison_inconsistency_metric(inconsistency_metric, num_ts, title):
    """
    Plot the value of an inconsistency metric for a time series for every SAX
    variant as a bar in a bar plot.

    :param inconsistency_metric: dict of len = 4
        Keys are the names of the four SAX variants. Values are pd.Series
        containing the value of an inconsistency metric for each evaluated
        time series for the respective SAX variant.
    :param num_ts: int
        The number of the time series in the pd.Series whose value of an
        inconsistency metric shall be plotted for each SAX variant.
    :param title: str
        The title the bar plot shall have.
    :return: None
    """

    sax_variants = list(inconsistency_metric.keys())
    values = [inconsistency_metric[sax_variant][num_ts] for sax_variant in sax_variants]

    plt.bar(sax_variants, values, width=0.4, edgecolor="black")
    plt.title(title)
    plt.show()
