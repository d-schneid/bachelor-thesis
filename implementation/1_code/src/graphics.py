import matplotlib.pyplot as plt


def plot_internal_clustering_metrics(silhouette, calinski_harabasz,
                                     davies_bouldin, num_clusters_min,
                                     ssq_error=None):
    """
    Plot the given internal evaluation metrics of a clustering.

    :param silhouette: np.array of shape (num_diff_clusters,)
        Contains the values of the Silhouette coefficient for each number of
        clusters.
    :param calinski_harabasz: np.array of shape (num_diff_clusterss,)
        Contains the values of the Calinski-Harabasz index for each number of
        clusters.
    :param davies_bouldin: np.array of shape (num_diff_clusters,)
        Contains the values of the Davies-Bouldin index for each number of
        clusters.
    :param num_clusters_min: float
        The lower bound (inclusive) of the number of clusters for that the
        evaluation of the clustering algorithm was run.
    :param ssq_error: np.array of shape (num_diff_clusters,) (default = None)
        Contains the values of the SSQ error for each number of clusters.
        If None, only the above three internal clustering metrics are plotted.
    :return:
        None
    """

    num_clusters_max = num_clusters_min + silhouette.size - 1
    x_ticks = range(num_clusters_min, num_clusters_max + 1)
    x_label = "number of clusters"
    plt.figure(figsize=(15, 5))

    position = 221
    plt.subplot(position)
    plt.plot(x_ticks, silhouette, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Silhouette")

    position = 222
    plt.subplot(position)
    plt.plot(x_ticks, calinski_harabasz, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Calinski-Harabasz")

    position = 223
    plt.subplot(position)
    plt.plot(x_ticks, davies_bouldin, "b-")
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.title("Davies-Bouldin")

    if ssq_error is not None:
        position = 224
        plt.subplot(position)
        plt.plot(x_ticks, ssq_error, "b-")
        plt.xticks(x_ticks)
        plt.xlabel(x_label)
        plt.title("SSQ-Error")

    plt.tight_layout()
    plt.show()
