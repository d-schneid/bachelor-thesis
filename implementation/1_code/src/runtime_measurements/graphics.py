import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_execution_times(execution_times, ts_sizes):
    """
    For each given number of time series points plot four points corresponding
    to the execution time in seconds for the respective SAX variants.

    :param execution_times: dict of len = 4
        Keys are the names of the four SAX variants. Values are lists
        containing the execution time in seconds for each evaluated number of
        time series points for the respective SAX variant.
    :param ts_sizes: list
        Contains the number of time series points in ascending order for that
        the execution times were measured.
    :return: None
    """

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    for sax_variant, execution_times in execution_times.items():
        ax.plot(execution_times, label=sax_variant)

    formatter = ticker.FuncFormatter(lambda x, pos: f"{x}s")
    ax.yaxis.set_major_formatter(formatter)

    plt.xticks([pos for pos in range(len(ts_sizes))], ts_sizes)
    plt.xlabel("Number of Time Series Points")

    plt.legend()
    plt.title("Execution Times in Seconds")
    plt.show()
