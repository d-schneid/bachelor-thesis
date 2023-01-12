import matplotlib.pyplot as plt


def highlight_motif(df_norm, start, end, idx_lst, num_col):
    """
    Highlight the points in a time series that belong to a motif in red.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset of which one time series along with
        a highlighted motif shall be plotted.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within its
        original time series
    :param end: np.array of shape (num_subsequences,)
        The index of the end (exclusive) of each subsequence within its
        original time series
    :param idx_lst: list of len(idx_lst) = num_subsequences_of_motif
        Contains the indexes of the start and end of the subsequences that
        are declared a motif and that shall be highlighted in their original
        time series. With such an index, the corresponding subsequence in the
        original time series can be queried with the help of 'start' and 'end'.
    :param num_col: int
        The number of the column (i.e. time series) in the given dataframe that
        shall be plotted with a highlighted motif.
    :return:
        None
    """

    plt.figure(figsize=(15, 5))
    plt.plot(df_norm.index, df_norm.iloc[:, num_col])

    for idx in idx_lst:
        df_highlight = df_norm.iloc[start[idx]:end[idx], num_col]
        plt.plot(df_highlight.index, df_highlight, "red")

    plt.show()
