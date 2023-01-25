from utils import constant_segmentation_overlapping
from pattern_recognition.motif_discovery.brute_force.shared import _do_brute_force


"""
The function below constitutes a brute force algorithm for motif discovery in
time series that is based on the 'Find-1-Motif-Brute-Force' algorithm in
reference [1]. The algorithm completely works on the raw values of a time
series.
The algorithm counts (non-trivial) matches for each subsequence of a fixed
length and reports the subsequence with the highest count along with the
corresponding matching subsequences. This is done iteratively such that the
reported motifs decrease in their number of similar subsequences contained.

References
----------
[1] Lonardi, J. L. E. K. S., & Patel, P. (2002, August). Finding motifs in time
series. In Proc. of the 2nd Workshop on Temporal Data Mining (pp. 53-68).
"""


def do_brute_force_raw(df_norm, len_subsequence, dist_threshold, num_diff_threshold,
                       p=1.0, ignore_trivial=False, exclusion_zone=1):
    """
    Execute a brute force algorithm for motif discovery in time series. This
    algorithm counts (non-trivial) matches for each subsequence of a fixed
    length and reports the subsequence with the highest count along with the
    corresponding matching subsequences. This is done iteratively such that the
    reported motifs decrease in their number of similar subsequences contained.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset for which motifs shall be found
        based on the brute force algorithm.
    :param len_subsequence: int
        The length the extracted subsequences of the time series shall have.
    :param dist_threshold: float
        The maximum distance based on the applied Minkowski distance such that
        two subsequences shall be considered similar.
    :param num_diff_threshold: int
        The maximum distance between any two points at the same position in two
        subsequences such that these two subsequences shall be considered
        similar.
    :param p: float (default = 1.0)
        The p-norm to apply for computing the Minkowski distance between two
        subsequences to check for their similarity.
    :param ignore_trivial: bool (default = False)
        If True: Trivial subsequence matches around 'exclusion_zone' (on the
        left and right side) of a subsequence are ignored for the motif they
        would be in, but not for further motifs.
    :param exclusion_zone: int (default = 1)
        The number of indexes on the left and right side of a match that shall
        be excluded from the motif.
        If 'ignore_trivial' is set to False, 'exclusion_zone' will be ignored.
    :return:
        motifs_lst: list of len(motifs_lst) = num_ts
            Contains a two-dimensional list for each time series.
            In this two-dimensional list, each sub-list corresponds to a motif
            pattern. Hence, each of these sub-lists contains the indexes of the
            start and end of the respective motifs belonging to the respective
            motif pattern in the respective original time series.
            With such an index of a motif, the corresponding subsequence in the
            original time series can be queried with the help of 'start'
            and 'end'.
        start: np.array of shape (num_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end: np.array of shape (num_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    """

    start, end = constant_segmentation_overlapping(df_norm.shape[0], len_subsequence, gap=1)
    motifs_lst = _do_brute_force(df_norm, start, end, p, dist_threshold, num_diff_threshold,
                                 ignore_trivial, exclusion_zone)

    return motifs_lst, start, end
