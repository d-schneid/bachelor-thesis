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
