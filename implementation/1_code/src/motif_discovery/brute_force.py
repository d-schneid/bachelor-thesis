from scipy.spatial import distance

from utils.utils import constant_segmentation_overlapping
from motif_discovery.utils import _get_linearized_encoded_sax, _remove_trivial


"""
The functions below constitute a brute force algorithm for motif discovery in
time series that is based on the 'Find-1-Motif-Brute-Force' algorithm in
reference [1]. The algorithm completely works on the respective encoded SAX
representation of a time series.
The algorithm counts (non-trivial) matches for each (encoded) subsequence of a
fixed length and reports the subsequence with the highest count along with the
corresponding matching subsequences. This is done iteratively such that the
reported motifs decrease in their number of similar subsequences contained.

References
----------
[1] Lonardi, J. L. E. K. S., & Patel, P. (2002, August). Finding motifs in time
series. In Proc. of the 2nd Workshop on Temporal Data Mining (pp. 53-68).
"""


def _hamming_distance(current_subsequence, subsequence):
    """
    Compute the Hamming distance between the two given subsequences.

    :param current_subsequence:
        pd.Series of shape (num_compare_segments * symbols_per_segment,)
            The encoded and linearized subsequence for that similar
            subsequences shall be found based on the Hamming distance.
    :param subsequence:
        pd.Series of shape (num_compare_segments * symbols_per_segment,)
            The encoded and linearized subsequence that shall be checked if it
            is similar to the 'current_subsequence' based on the Hamming
            distance.
    :return: int
        The Hamming distance between the 'current_subsequence' and the
        'subsequence' as the number of encoded symbols (i.e. integer numbers)
        that differ between them.
    """

    return sum(fst_num != snd_num
               for fst_num, snd_num in zip(current_subsequence, subsequence))


def _too_large_num_diff(current_subsequence, subsequence, num_diff_threshold):
    """
    Check if the difference between any two encoded symbols at the same
    position in the two given subsequences is too large based on the given
    threshold.

    :param current_subsequence:
        pd.Series of shape (num_compare_segments * symbols_per_segment,)
            The encoded and linearized subsequence for that similar
            subsequences shall be found based on this check.
    :param subsequence:
        pd.Series of shape (num_compare_segments * symbols_per_segment,)
            The encoded and linearized subsequence that shall be checked if it
            is similar to the 'current_subsequence' based on this check.
    :param num_diff_threshold: int
        The maximum absolute difference between any two encoded symbols at the
        same position in 'current_subsequence' and 'subsequence' such that
        these two given subsequences are considered similar based on this
        check.
    :return: bool
        True: If there are two encoded symbols at the same position in
        'current_subsequence' and 'subsequence' whose absolute difference is
        larger than the given threshold 'num_diff_threshold'.
        False: If any two encoded symbols at the same position in
        'current_subsequence' and 'subsequence' have a absolute difference of
        at most the given threshold 'num_diff_threshold'.
    """

    for fst_num, snd_num in zip(current_subsequence, subsequence):
        if abs(fst_num - snd_num) > num_diff_threshold:
            return True
    return False


def _find_best_motif(encoded_sax_subsequences, assigned, p, dist_threshold,
                     hamming_threshold, num_diff_threshold, ignore_trivial,
                     exclusion_zone):
    """
    Iterate over all available subsequences and count the respective similar
    subsequences. Declare the set with the highest count as the best available
    motif.

    :param encoded_sax_subsequences: list of len = num_subsequences
        Contains all extracted subsequences of a linearized encoded SAX
        representation as a pd.Series of shape
        (num_compare_segments * symbols_per_segment,).
    :param assigned: set
        Contains the indexes of all subsequences that were already assigned to
        a motif in a previous round.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between two
        encoded subsequences to check for their similarity.
    :param dist_threshold: float
        The maximum distance based on the applied Minkowski distance such that
        two encoded subsequences shall be considered similar.
    :param hamming_threshold: int
        The maximum distance based on the Hamming distance such that two
        encoded subsequences shall be considered similar.
        The Hamming distance on the encoded subsequences has the same
        interpretation as the Hamming distance on the symbolic representations
        of the subsequences.
    :param num_diff_threshold: int
        The maximum distance between any two encoded symbols at the same
        position in two subsequences such that these two subsequences shall be
        considered similar.
    :param ignore_trivial: bool
        If True: Trivial subsequence matches around 'exclusion_zone' (on the
        left and right side) of a subsequence are ignored for the motif they
        would be in, but not for further motifs.
    :param exclusion_zone: int
        The number of indexes on the left and right side of a match that shall
        be excluded from the motif.
        If 'ignore_trivial' is set to False, 'exclusion_zone' will be ignored.
    :return: list
        Contains the indexes of all subsequences that are similar to a
        subsequence (inclusive) that has the highest count of (non-trivial)
        matches.
    """

    best_count = 0
    best_motif = []
    initial = [(idx, subsequence)
               for idx, subsequence in enumerate(encoded_sax_subsequences)
               if idx not in assigned]

    for current_subsequence in encoded_sax_subsequences:
        # will contain self-match
        motif = []
        filtered = [(idx, subsequence) for idx, subsequence in initial
                    if distance.minkowski(current_subsequence, subsequence, p) <= dist_threshold]
        filtered = [(idx, subsequence) for idx, subsequence in filtered
                    if _hamming_distance(current_subsequence, subsequence) <= hamming_threshold]
        filtered = [(idx, subsequence) for idx, subsequence in filtered
                    if not _too_large_num_diff(current_subsequence, subsequence, num_diff_threshold)]

        motif.extend([idx for idx, subs in filtered])
        if ignore_trivial:
            motif, trivial = _remove_trivial(motif, exclusion_zone)

        if len(motif) > best_count:
            best_count = len(motif)
            best_motif = motif

    return best_motif


def do_brute_force(df_norm, window_size, sax_variant, num_compare_segments,
                   dist_threshold, hamming_threshold, num_diff_threshold,
                   p=1.0, ignore_trivial=False, exclusion_zone=1):
    """
    Execute a brute force algorithm for motif discovery in time series. This
    algorithm counts (non-trivial) matches for each (encoded) subsequence of a
    fixed length and reports the subsequence with the highest count along with
    the corresponding matching subsequences. This is done iteratively such that
    the reported motifs decrease in their number of similar subsequences
    contained.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset for which motifs shall be found
        based on the brute force algorithm.
    :param window_size: int
        The size of the window that is used to transform each time series into
        its PAA representation.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the PAA representation
        of a time series into its symbolic representation.
    :param num_compare_segments: int
        The length the extracted subsequences of the SAX representation shall
        have expressed in the number of segments of the SAX representation.
    :param p: float (default = 1.0)
        The p-norm to apply for computing the Minkowski distance between two
        encoded subsequences to check for their similarity.
    :param dist_threshold: float
        The maximum distance based on the applied Minkowski distance such that
        two encoded subsequences shall be considered similar.
    :param hamming_threshold: int
        The maximum distance based on the Hamming distance such that two
        encoded subsequences shall be considered similar.
        The Hamming distance on the encoded subsequences has the same
        interpretation as the Hamming distance on the symbolic representations
        of the subsequences.
    :param num_diff_threshold: int
        The maximum distance between any two encoded symbols at the same
        position in two subsequences such that these two subsequences shall be
        considered similar.
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
            original time series can be queried with the help of
            'start' and 'end'.
        start:
            np.array of shape (num_encoded_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end:
            np.array of shape (num_encoded_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    """

    df_sax_linearized_encoded, df_sax = _get_linearized_encoded_sax(df_norm, window_size, sax_variant)

    # use gap to account for multiple symbols per segment
    start, end = constant_segmentation_overlapping(df_sax_linearized_encoded.shape[0],
                                                   num_compare_segments * sax_variant.symbols_per_segment,
                                                   sax_variant.symbols_per_segment)

    ts_split = []
    for i in range(df_sax_linearized_encoded.shape[1]):
        ts_split.append(df_sax_linearized_encoded.iloc[:, i])

    ts_subsequences = []
    num_subsequences = len(start)
    for encoded_sax_word in ts_split:
        subsequences = [encoded_sax_word.iloc[start[i]:end[i]]
                        for i in range(num_subsequences)]
        ts_subsequences.append(subsequences)

    motifs_lst = []
    for encoded_sax_subsequences in ts_subsequences:
        ts_motifs = []
        # indexes that are already assigned to a motif
        assigned = set()
        while True:
            best_motif = _find_best_motif(encoded_sax_subsequences, assigned,
                                          p, dist_threshold, hamming_threshold,
                                          num_diff_threshold, ignore_trivial,
                                          exclusion_zone)
            if len(best_motif) <= 1:
                break
            ts_motifs.append(best_motif)
            # motifs shall be disjoint
            assigned.update(best_motif)

        motifs_lst.append(ts_motifs)

    # adjust for number of symbols per segment and discretization
    start = (start // sax_variant.symbols_per_segment) * window_size
    end = (end // sax_variant.symbols_per_segment) * window_size
    return motifs_lst, start, end
