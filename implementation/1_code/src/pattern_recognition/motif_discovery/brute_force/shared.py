from scipy.spatial import distance

from pattern_recognition.motif_discovery.utils import _remove_trivial


def _hamming_distance(current_subsequence, subsequence):
    """
    Compute the Hamming distance between the two given subsequences.

    :param current_subsequence:
        pd.Series of shape (len_subsequence,)
            The subsequence for that similar subsequences shall be found based
            on the Hamming distance.
    :param subsequence:
        pd.Series of shape (len_susequence,)
            The subsequence that shall be checked if it is similar to the
            'current_subsequence' based on the Hamming distance.
    :return: int
    """

    return sum(fst_num != snd_num
               for fst_num, snd_num in zip(current_subsequence, subsequence))


def _too_large_num_diff(current_subsequence, subsequence, num_diff_threshold):
    """
    Check if the difference between any two points at the same position in the
    two given subsequences is too large based on the given threshold.

    :param current_subsequence:
        pd.Series of shape (len_subsequence,)
            The subsequence for that similar subsequences shall be found based
            on this check.
    :param subsequence:
        pd.Series of shape (len_subsequence,)
            The subsequence that shall be checked if it is similar to the
            'current_subsequence' based on this check.
    :param num_diff_threshold: int
        The maximum absolute difference between any two points at the same
        position in 'current_subsequence' and 'subsequence' such that these two
        given subsequences are considered similar based on this check.
    :return: bool
        True: If there are two points at the same position in
        'current_subsequence' and 'subsequence' whose absolute difference is
        larger than the given threshold 'num_diff_threshold'.
        False: If any two points at the same position in 'current_subsequence'
        and 'subsequence' have an absolute difference of at most the given
        threshold 'num_diff_threshold'.
    """

    for fst_num, snd_num in zip(current_subsequence, subsequence):
        if abs(fst_num - snd_num) > num_diff_threshold:
            return True
    return False


def _find_best_motif(subsequences, assigned, dist_threshold, num_diff_threshold,
                     p, ignore_trivial, exclusion_zone, hamming_threshold=None):
    """
    Iterate over all available subsequences and count the respective similar
    subsequences. Declare the set with the highest count as the best available
    motif.

    :param subsequences: list of len = num_subsequences
        Contains all extracted subsequences as a pd.Series of shape
        (len_subsequence,).
    :param assigned: set
        Contains the indexes of all subsequences that were already assigned to
        a motif in a previous round.
    :param dist_threshold: float
        The maximum distance based on the applied Minkowski distance such that
        two subsequences shall be considered similar.
    :param num_diff_threshold: int
        The maximum distance between any two points at the same position in two
        subsequences such that these two subsequences shall be considered
        similar.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between two
        subsequences to check for their similarity.
    :param ignore_trivial: bool
        If True: Trivial subsequence matches around 'exclusion_zone' (on the
        left and right side) of a subsequence are ignored for the motif they
        would be in, but not for further motifs.
    :param exclusion_zone: int
        The number of indexes on the left and right side of a match that shall
        be excluded from the motif.
        If 'ignore_trivial' is set to False, 'exclusion_zone' will be ignored.
    :param hamming_threshold: int (default = None)
        This is only used for the discretized version of the brute force
        algorithm, since it does not make sense for the raw version.
        The maximum distance based on the Hamming distance such that two
        encoded subsequences shall be considered similar.
        The Hamming distance on the encoded subsequences has the same
        interpretation as the Hamming distance on the symbolic representations
        of the subsequences.
    :return: list
        Contains the indexes of all subsequences that are similar to a
        subsequence (inclusive) that has the highest count of (non-trivial)
        matches.
    """

    best_count = 0
    best_motif = []
    initial = [(idx, subsequence)
               for idx, subsequence in enumerate(subsequences)
               if idx not in assigned]

    for current_subsequence in subsequences:
        # will contain self-match
        motif = []
        filtered = [(idx, subsequence) for idx, subsequence in initial
                    if distance.minkowski(current_subsequence, subsequence, p) <= dist_threshold]
        filtered = [(idx, subsequence) for idx, subsequence in filtered
                    if not _too_large_num_diff(current_subsequence, subsequence, num_diff_threshold)]
        if hamming_threshold is not None:
            filtered = [(idx, subsequence) for idx, subsequence in filtered
                        if _hamming_distance(current_subsequence, subsequence) <= hamming_threshold]

        motif.extend([idx for idx, subsequence in filtered])
        if ignore_trivial:
            motif, trivial = _remove_trivial(motif, exclusion_zone)

        if len(motif) > best_count:
            best_count = len(motif)
            best_motif = motif

    return best_motif


def _do_brute_force(df_ts, start, end, p, dist_threshold, num_diff_threshold,
                    ignore_trivial, exclusion_zone, hamming_threshold=None):
    """
    Compute the motifs for the given time series.

    :param df_ts: dataframe of shape (ts_size, num_ts)
        The time series for that motifs shall be found.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within its
        original time series
    :param end: np.array of shape (num_subsequences,)
        The index of the end (exclusive) of each subsequence within its
        original time series.
    :param p: (default = 1.0)
        The p-norm to apply for computing the Minkowski distance between two
        subsequences to check for their similarity.
    :param dist_threshold: float
        The maximum distance based on the applied Minkowski distance such that
        two subsequences shall be considered similar.
    :param num_diff_threshold: int
        The maximum distance between any two points at the same position in two
        subsequences such that these two subsequences shall be considered
        similar.
    :param ignore_trivial: bool (default = False)
        If True: Trivial subsequence matches around 'exclusion_zone' (on the
        left and right side) of a subsequence are ignored for the motif they
        would be in, but not for further motifs.
    :param exclusion_zone: int (default = 1)
        The number of indexes on the left and right side of a match that shall
        be excluded from the motif.
        If 'ignore_trivial' is set to False, 'exclusion_zone' will be ignored.
    :param hamming_threshold: int (default = None)
        This is only used for the discretized version of the brute force
        algorithm, since it does not make sense for the raw version.
        The maximum distance based on the Hamming distance such that two
        encoded subsequences shall be considered similar.
        The Hamming distance on the encoded subsequences has the same
        interpretation as the Hamming distance on the symbolic representations
        of the subsequences.
    :return: list of len(motifs_lst) = num_ts
        Contains a two-dimensional list for each time series.
        In this two-dimensional list, each sub-list corresponds to a motif
        pattern. Hence, each of these sub-lists contains the indexes of the
        start and end of the respective motifs belonging to the respective
        motif pattern in the respective original time series.
        With such an index of a motif, the corresponding subsequence in the
        original time series can be queried with the help of 'start' and 'end'.
    """

    ts_split = []
    for i in range(df_ts.shape[1]):
        ts_split.append(df_ts.iloc[:, i])

    ts_subsequences = []
    num_subsequences = len(start)
    for ts in ts_split:
        subsequences = [ts.iloc[start[i]:end[i]]
                        for i in range(num_subsequences)]
        ts_subsequences.append(subsequences)

    motifs_lst = []
    for subsequences in ts_subsequences:
        ts_motifs = []
        # indexes that are already assigned to a motif
        assigned = set()
        while True:
            best_motif = _find_best_motif(subsequences, assigned, dist_threshold,
                                          num_diff_threshold, p, ignore_trivial,
                                          exclusion_zone, hamming_threshold)
            if len(best_motif) <= 1:
                break
            ts_motifs.append(best_motif)
            # motifs shall be disjoint
            assigned.update(best_motif)

        motifs_lst.append(ts_motifs)

    return motifs_lst
