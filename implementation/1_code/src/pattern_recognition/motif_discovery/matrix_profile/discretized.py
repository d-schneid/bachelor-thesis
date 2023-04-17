import stumpy
import numpy as np
from scipy.spatial import distance

from utils import constant_segmentation_overlapping
from pattern_recognition.motif_discovery.utils import _encode_symbols
from pattern_recognition.utils import get_linearized_encoded_sax
from pattern_recognition.motif_discovery.matrix_profile.shared import _find_nearest_neighs
from pattern_recognition.motif_discovery.utils import _remove_trivial


"""
The functions below constitute an algorithm for motif discovery in time series
that is based on the Matrix Profile. The Matrix Profile is applied on the
symbolic representations of time series based on the respective SAX variant
that is used. Since the Matrix Profile requires numerical values, the symbols
of these representations are encoded into numbers.
Moreover, the idea of finding similar subsequences in a bounded area around
two selected and promising subsequences from reference [1] is used to discover
motifs.

References
----------
[1] Chiu, Bill, Eamonn Keogh, and Stefano Lonardi. "Probabilistic discovery of
time series motifs." Proceedings of the ninth ACM SIGKDD international
conference on Knowledge discovery and data mining. 2003. (expanded version)
"""


def _compute_nearest_neighs(sax_word_split_encoded, num_compare_segments,
                            max_distance, normalize, p):
    """
    Compute possible motifs based on the nearest neighbor of each subsequence
    computed by the Matrix Profile. This is done for each split of the SAX word
    based on the number of symbols per segment. As a result, the motifs from
    each split are merged.

    :param sax_word_split_encoded:
        list of len(sax_word_split_encoded) = number of symbols per segment
        Contains an encoded dataframe for each split based on the number of
        symbols per segment.
    :param num_compare_segments: int
        The window size to extract subsequences. In the encoded case, this
        corresponds to the number of segments that shall be extracted and
        compared to each other.
    :param max_distance: float
        The upper bound for the distance between a subsequence and a query,
        such that the subsequence is considered similar to the query.
    :param normalize: bool
        True if the encoded SAX words shall be z-normalized before applying the
        Matrix Profile.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between a
        subsequence and a query to check for their similarity.
    :return:
        A dictionary that contains tuples with the indexes of two nearest
        neighbors as keys and the corresponding distance as values.
        The subsequences corresponding to the contained indexes are considered
        as a possible motif.
    """

    matrix_profiles = []
    for sax_word_encoded in sax_word_split_encoded:
        # adjust format for Matrix Profile
        sax_word_encoded = np.array(sax_word_encoded.iloc[:, 0]).astype(np.float64)
        # contains nearest neighbors with respective distances
        matrix_profile = stumpy.stump(sax_word_encoded, m=num_compare_segments, normalize=normalize, p=p)[:, :2]
        matrix_profiles.append(matrix_profile)

    nearest_neighs_lst = []
    for matrix_profile in matrix_profiles:
        nearest_neighs = _find_nearest_neighs(matrix_profile, max_distance)
        nearest_neighs_lst.append(nearest_neighs)

    # for overlapping possible motifs, take those with the higher distance,
    # because this is a lower bound for the distance between these possible
    # motifs on the full SAX word
    merged_nearest_neighs = {nearest_neighs_idxs: max(nearest_neighs.get(nearest_neighs_idxs, 0.0)
                                                      for nearest_neighs in nearest_neighs_lst)
                             for nearest_neighs_idxs in set().union(*nearest_neighs_lst)}

    # nearest neighbors with the smallest distance first in order to retrieve
    # most promising subsequences for a motif first
    return dict(sorted(merged_nearest_neighs.items(), key=lambda tup: tup[1]))


def _check_distances(sax_word_linearized_encoded, fst_neigh_linearized_encoded,
                     snd_neigh_linearized_encoded, fst_motif_idxs, snd_motif_idxs,
                     start_linearized, end_linearized, max_distance, p):
    """
    Build the motif of 'fst_neigh_linearized_encoded' and
    'snd_neigh_linearized_encoded' by checking the distances between them and
    their potential similar subsequences.

    :param sax_word_linearized_encoded:
        pd.Series of shape (num_segments * num_symbols_per_segment,)
        The linearized and encoded SAX representation in that motifs shall be
        found.
    :param fst_neigh_linearized_encoded:
        pd.Series of shape (num_compare_segments * num_symols_per_segment,)
        The first of the two nearest neighbors that form a motif.
    :param snd_neigh_linearized_encoded:
        pd.Series of shape (num_compare_segments * num_symols_per_segment,)
        The second of the two nearest neighbors that form a motif.
    :param fst_motif_idxs: list
        Contains the indexes of all subsequences across all symbolic positions
        that are within 'max_distance' of the first nearest neighbor.
    :param snd_motif_idxs: list
        Contains the indexes of all subsequences across all symbolic positions
        that are within 'max_distance' of the first nearest neighbor.
    :param start_linearized: np.array of shape (num_encoded_subsequences,)
        The index of the start (inclusive) of each encoded and linearized
        subsequence within its encoded and linearized SAX representation.
    :param end_linearized: np.array of shape (num_encoded_subsequences,)
        The index of the end (exclusive) of each encoded and linearized
        subsequence within its encoded and linearized SAX representation.
    :param max_distance: float
        The maximum distance between 'fst_neigh_linearized_encoded' or
        'snd_neigh_linearized_encoded' with an encoded subsequence, such that
        the encoded subsequence is considered similar.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between the
        two subsequences to check for their similarity.
    :return: list
        Contains the indexes of the start and end of the respective
        subsequences within the given time series that belong to the motif of
        'fst_neigh_linearized_encoded' and 'snd_neigh_linearized_encoded'.
        With such an index, the corresponding subsequence in the original time
        series can be queried with the help of 'start' and 'end'.
    """

    # union, because all these subsequences are within 'max_distance' for at
    # least one of the two nearest neighbors and for at least one symbolic
    # position based on the number of symbols per segment
    possible_motifs_idxs = list(set(fst_motif_idxs) | set(snd_motif_idxs))
    motif = []
    # find true motifs by considering 'max_distance' on full linearized SAX
    # words
    for idx in possible_motifs_idxs:
        possible_motif_linearized_encoded = sax_word_linearized_encoded.iloc[start_linearized[idx]:end_linearized[idx]]
        fst_dist = distance.minkowski(fst_neigh_linearized_encoded, possible_motif_linearized_encoded, p=p)
        if fst_dist <= max_distance:
            motif.append(idx)
        # avoid computing second distance if not necessary
        else:
            snd_dist = distance.minkowski(snd_neigh_linearized_encoded, possible_motif_linearized_encoded, p=p)
            # true motif is within 'max_distance' of at least one of the two
            # given nearest neighbors
            if fst_dist <= max_distance or snd_dist <= max_distance:
                motif.append(idx)

    return motif


def _build_motif(sax_word_split_encoded, idxs, start, end, max_distance,
                 sax_word_linearized_encoded, fst_neigh_linearized_encoded,
                 snd_neigh_linearized_encoded, start_linearized,
                 end_linearized, normalize, p):
    """
    Find all similar subsequences that are within 'max_distance' compared to
    at least one of the two given 'fst_neigh_linearized_encoded' and
    'snd_neigh_linearized_encoded'. These similar subsequences belong to the
    motif of 'fst_neigh_linearized_encoded' and 'snd_neigh_linearized_encoded'.

    :param sax_word_split_encoded:
        list of len(sax_word_split_encoded) = number of symbols per segment
        Contains an encoded dataframe for each split based on the number of
        symbols per segment.
    :param idxs: tuple
        Contains the indexes of 'fst_neigh_linearized_encoded' and
        'snd_neigh_linearized_encoded' based on the given 'start' and 'end'.
    :param start: np.array of shape (num_encoded_subsequences,)
        The index of the start (inclusive) of each encoded subsequence within
        its encoded SAX representation.
    :param end: np.array of shape (num_encoded_subsequences,)
        The index of the end (exclusive) of each encoded subsequence within its
        encoded SAX representation.
    :param max_distance: float
        The maximum distance between 'fst_neigh_linearized_encoded' or
        'snd_neigh_linearized_encoded' with an encoded subsequence, such that
        the encoded subsequence is considered similar.
    :param sax_word_linearized_encoded:
        pd.Series of shape (num_segments * num_symbols_per_segment,)
        The linearized and encoded SAX representation in that motifs for
        'fst_neigh_linearized_encoded' and 'snd_neigh_linearized_encoded' shall
        be found.
    :param fst_neigh_linearized_encoded:
        pd.Series of shape (num_compare_segments * num_symbols_per_segment,)
        The first neighbor for that similar subsequences in
        'sax_word_linearized_encoded' shall be found.
    :param snd_neigh_linearized_encoded:
        pd.Series of shape (num_compare_segments * num_symbols_per_segment,)
        The second neighbor for that similar subsequences in
        'sax_word_linearized_encoded' shall be found.
    :param start_linearized: np.array of shape (num_encoded_subsequences,)
        The index of the start (inclusive) of each encoded and linearized
        subsequence within its encoded and linearized SAX representation.
    :param end_linearized: np.array of shape (num_encoded_subsequences,)
        The index of the end (exclusive) of each encoded and linearized
        subsequence within its encoded and linearized SAX representation.
    :param normalize: bool
        True if the nearest neighbors and the queried time series shall be
        z-normalized before computing distances.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between the
        two given nearest neighbors and subsequences to check for their
        similarity.
    :return:
        A list that contains the starting indexes of the subsequences that
        belong to the motif of the two given nearest neighbors.
    """

    fst_motif_idxs, snd_motif_idxs = [], []
    for idx, sax_word_encoded in enumerate(sax_word_split_encoded):
        sax_word_encoded = np.array(sax_word_encoded.iloc[:, 0]).astype(np.float64)

        fst_neigh = sax_word_encoded[start[idxs[0]]:end[idxs[0]]]
        # indexes of all subsequences within 'sax_word_encoded' that are within
        # 'max_distance' of the 'fst_neigh'; self-match is contained as well
        fst_motif_idxs.extend(stumpy.match(fst_neigh, sax_word_encoded, max_distance=max_distance,
                                           normalize=normalize, p=p)[:, 1])

        snd_neigh = sax_word_encoded[start[idxs[1]]:end[idxs[1]]]
        snd_motif_idxs.extend(stumpy.match(snd_neigh, sax_word_encoded, max_distance=max_distance,
                                           normalize=normalize, p=p)[:, 1])

    motif = _check_distances(sax_word_linearized_encoded, fst_neigh_linearized_encoded,
                             snd_neigh_linearized_encoded, fst_motif_idxs, snd_motif_idxs,
                             start_linearized, end_linearized, max_distance, p)
    return motif


def _build_ts_motifs(sax_word_linearized_encoded, sax_word_split_encoded,
                     sax_variant, nearest_neighs, start, end, max_distance,
                     normalize, p, exclusion_zone):
    """
    Compute all motifs of the given SAX representation.

    :param sax_word_linearized_encoded:
        pd.Series of shape (num_segments * num_symbols_per_segment,)
        The linearized and encoded SAX representation in that motifs shall be
        found.
    :param sax_word_split_encoded:
        list of len(sax_word_split_encoded) = number of symbols per segment
        Contains an encoded dataframe for each split based on the number of
        symbols per segment.
    :param sax_variant: AbstractSAX
        The SAX variant that was used to create the given SAX representations.
    :param nearest_neighs: dict
        Contains tuples with the indexes of two nearest neighbors as keys and
        the corresponding distance as values. The subsequences corresponding to
        the contained indexes are considered as a possible motif.
    :param start: np.array of shape (num_encoded_subsequences,)
        The index of the start (inclusive) of each subsequence within its
        original time series.
    :param end: np.array of shape (num_encoded_subsequences,)
        The index of the end (exclusive) of each subsequence within its
        original time series.
    :param max_distance: float
        The maximum distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :param normalize: bool (default = False)
        True if the distances between two subsequences shall be computed on
        z-normalized versions of them.
        False otherwise.
    :param p: float (default = 1.0)
        The p-norm to apply for computing the Minkowski distance between two
        encoded subsequences to check for their similarity.
    :return: list
        Contains a list for each found motif in the given SAX representation.
        Each of these sub-lists contains the indexes of the start and end of
        the respective subsequences within the given SAX representation.
        With such an index, the corresponding subsequence in the original time
        series can be queried with the help of 'start' and 'end'.
    """

    # indexes for extraction of linearized encoded subsequences
    start_linearized = start * sax_variant.symbols_per_segment
    end_linearized = end * sax_variant.symbols_per_segment
    nearest_neighs_idxs = list(nearest_neighs.keys())
    ts_motifs = []
    # indexes that are already assigned to a motif
    assigned = set()
    for idxs in nearest_neighs_idxs:
        # motifs are disjoint
        # at least one of the two current subsequences already within
        # 'max_distance' of a previously encountered subsequence
        if idxs[0] in assigned or idxs[1] in assigned:
            continue

        fst_neigh_linearized_encoded = sax_word_linearized_encoded.iloc[start_linearized[idxs[0]]:end_linearized[idxs[0]]]
        snd_neigh_linearized_encoded = sax_word_linearized_encoded.iloc[start_linearized[idxs[1]]:end_linearized[idxs[1]]]
        dist_neighs = distance.minkowski(fst_neigh_linearized_encoded, snd_neigh_linearized_encoded, p=float(p))
        # check if the two nearest neighbors are within 'max_distance' of
        # each other, and therefore form a motif
        if dist_neighs > max_distance:
            continue

        # find all subsequences (respectively their indexes) that are
        # within 'max_distance' of at least one of the two nearest neighbors
        # these subsequences belong to the motif of the two nearest neighbors
        motif = _build_motif(sax_word_split_encoded, idxs, start, end, float(max_distance),
                             sax_word_linearized_encoded, fst_neigh_linearized_encoded,
                             snd_neigh_linearized_encoded, start_linearized,
                             end_linearized, normalize, float(p))
        motif.sort()
        motif, trivial = _remove_trivial(motif, exclusion_zone)
        assigned.update(trivial)
        if motif:
            ts_motifs.append(motif)
            assigned.update(motif)

    return ts_motifs


def do_matrix_profile_discretized(df_norm, window_size, sax_variant, num_compare_segments,
                                  max_distance, exclusion_zone, normalize=False, p=1.0, gap=1):
    """
    Execute the algorithm based on the computation of the Matrix Profile for
    motif discovery in a time series based on the subsequences of its symbolic
    representation.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset for which motifs shall be found
        based on their symbolic representations.
    :param window_size: int
        The size of the window that is used to transform each time series into
        its PAA representation.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the PAA representation
        of a time series into its symbolic representation.
    :param num_compare_segments: int
        The length the extracted subsequences of the SAX representation shall
        have expressed in the number of segments of the SAX representation.
    :param max_distance: float
        The maximum distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :param normalize: bool (default = False)
        True if the distances between two subsequences shall be computed on
        z-normalized versions of them.
        False otherwise.
    :param p: float (default = 1.0)
        The p-norm to apply for computing the Minkowski distance between two
        encoded subsequences to check for their similarity.
    :param gap: int (default = 1)
        The gap between two consecutive encoded subsequences within the
        corresponding encoded SAX representation when extracting them.
    :return:
        motifs_lst: list of len(motifs_lst) = num_ts
            Contains a two-dimensional list for each time series.
            In this two-dimensional list, each sub-list corresponds to a motif
            pattern. Hence, each of these sub-lists contains the indexes of the
            start and end of the respective motifs belonging to the respective
            motif pattern in the respective original time series.
            With such an index of a motif, the corresponding subsequence in the
            original time series can be queried with the help of
            'adjusted_start' and 'adjusted_end'.
        start * window_size (adjusted_start):
            np.array of shape (num_encoded_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end * window_size (adjusted_end):
            np.array of shape (num_encoded_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    :raises:
        ValueError: If 'window_size' > ts_size.
        ValueError: If 'num_compare_segments' > num_segments.
    """

    df_sax_linearized_encoded, df_sax = get_linearized_encoded_sax(df_norm, window_size, sax_variant)
    # split SAX word based on number of symbols per segment
    # for each time series, one dataframe per number of symbols per segment
    symbol_split = []
    for col_name, col_sax_word in df_sax.items():
        col_sax_word = col_sax_word.to_frame()
        symbol_split.append([col_sax_word.applymap(lambda symbols: symbols[i])
                             for i in range(sax_variant.symbols_per_segment)])
    symbol_split_encoded = _encode_symbols(symbol_split, sax_variant)

    num_segments = df_sax.shape[0]
    # indexes for extraction of encoded subsequences
    start, end = constant_segmentation_overlapping(num_segments, num_compare_segments, gap)
    motifs_lst = []
    for idx, sax_word_split_encoded in enumerate(symbol_split_encoded):
        # find nearest neighbor for each extracted subsequence per number of
        # symbols per segment along with the corresponding distance
        # nearest neighbors are declared as potential motifs
        nearest_neighs = _compute_nearest_neighs(sax_word_split_encoded, num_compare_segments, float(max_distance), normalize, float(p))
        sax_word_linearized_encoded = df_sax_linearized_encoded.iloc[:, idx]
        ts_motifs = _build_ts_motifs(sax_word_linearized_encoded, sax_word_split_encoded,
                                     sax_variant, nearest_neighs, start, end,
                                     max_distance, normalize, p, exclusion_zone)
        motifs_lst.append(ts_motifs)

    # adjust start and end for indexes in original time series
    return motifs_lst, start * window_size, end * window_size
