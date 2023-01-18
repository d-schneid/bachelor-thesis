import stumpy
import numpy as np
from scipy.spatial import distance

from utils.utils import constant_segmentation_overlapping
from motif_discovery.utils import _get_linearized_encoded_sax, _encode_symbols


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


def _compute_possible_motifs(sax_word_split_encoded, num_compare_segments,
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
        matrix_profile = stumpy.stump(sax_word_encoded, m=num_compare_segments,
                                      normalize=normalize, p=p)[:, :2]
        matrix_profiles.append(matrix_profile)

    possible_motifs_lst = []
    for matrix_profile in matrix_profiles:
        possible_motifs = {}
        for idx, subsequence in enumerate(matrix_profile):
            nearest_neighbor = matrix_profile[subsequence[1]]
            # 'nearest_neighbor' of 'subsequence' has another neighbor with
            # that it has a lower distance than to 'subsequence'
            # not all subsequences have reciprocal nearest neighbors, because
            # being a nearest neighbor is not symmetric
            # therefore, avoid checking pairs that are not reciprocal nearest
            # neighbors
            # also, avoid checking reciprocal nearest neighbors twice
            if (nearest_neighbor[1], subsequence[1]) in possible_motifs or\
                    (subsequence[1], nearest_neighbor[1]) in possible_motifs:
                continue

            # reciprocal nearest neighbors
            if nearest_neighbor[1] == idx:
                # reciprocal nearest neighbors are within 'max_distance' of
                # each other
                # distance is symmetric: subsequence[0] == nearest_neighbor[0]
                if subsequence[0] <= max_distance:
                    possible_motifs[(idx, subsequence[1])] = subsequence[0]

        possible_motifs_lst.append(possible_motifs)

    # for overlapping possible motifs, take those with the higher distance,
    # because this is a lower bound for the distance between these possible
    # motifs on the full SAX word
    merged_possible_motifs = {nearest_neighbors: max(possible_motifs.get(nearest_neighbors, 0.0)
                                                     for possible_motifs in possible_motifs_lst)
                              for nearest_neighbors in set().union(*possible_motifs_lst)}

    # nearest neighbors with the smallest distance first in order to retrieve
    # most promising subsequences for a motif first
    return dict(sorted(merged_possible_motifs.items(), key=lambda tup: tup[1]))


def _find_motif(query, sax_word_encoded, max_distance, normalize, p):
    """
    Find the indexes of all subsequences within 'sax_word_encoded' that are
    within 'max_distance' of the 'query'. The self-match will be found as well.

    :param query: np.array of shape (num_compare_segments,)
        The query sequence for that similar subsequences shall be found in
        'sax_word_encoded'.
    :param sax_word_encoded: np.array of shape (num_segments,)
        The time series in that similar subsequences of the 'query' shall be
        found.
    :param max_distance: float
        The maximum distance between the 'query' and a subsequence from
        'sax_word_encoded' such that the subsequence is considered similar.
    :param normalize: bool
        True if the 'query' and 'sax_word_encoded' shall be z-normalized before
        computing distances.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between a
        subsequence from 'sax_word_encoded' and the 'query' to check for their
        similarity.
    :return:
        A list with the start indexes of the subsequences from
        'sax_word_encoded' that are considered similar to the given 'query'.
    """

    return stumpy.match(query, sax_word_encoded, max_distance=max_distance,
                        normalize=normalize, p=p)[:, 1]


def _build_motif(sax_word_split_encoded, idxs, start, end, max_distance,
                 sax_word_linearized_encoded, fst_query_linearized_encoded,
                 snd_query_linearized_encoded, start_linearized,
                 end_linearized, normalize, p):
    """
    Find all similar subsequences that are within 'max_distance' compared to
    at least one of the two given 'fst_query_linearized_encoded' and
    'snd_query_linearized_encoded'. These similar subsequences belong to the
    motif of 'fst_query_linearized_encoded' and 'snd_query_linearized_encoded'.

    :param sax_word_split_encoded:
        list of len(sax_word_split_encoded) = number of symbols per segment
        Contains an encoded dataframe for each split based on the number of
        symbols per segment.
    :param idxs: tuple
        Contains the indexes of 'fst_query_linearized_encoded' and
        'snd_query_linearized_encoded' based on the given 'start' and 'end'.
    :param start: np.array of shape (num_encoded_subsequences,)
        The index of the start (inclusive) of each encoded subsequence within
        its encoded SAX representation.
    :param end: np.array of shape (num_encoded_subsequences,)
        The index of the end (exclusive) of each encoded subsequence within its
        encoded SAX representation.
    :param max_distance: float
        The maximum distance between 'fst_query_linearized_encoded' or
        'snd_query_linearized_encoded' with an encoded subsequence, such that
        the encoded subsequence is considered similar.
    :param sax_word_linearized_encoded:
        pd.Series of shape (num_segments * num_symbols_per_segment,)
        The linearized and encoded SAX representation in that motifs for
        'fst_query_linearized_encoded' and 'snd_query_linearized_encoded' shall
        be found.
    :param fst_query_linearized_encoded:
        pd.Series of shape (num_compare_segments * num_symbols_per_segment,)
        The first query for that similar subsequences in
        'sax_word_linearized_encoded' shall be found.
    :param snd_query_linearized_encoded:
        pd.Series of shape (num_compare_segments * num_symbols_per_segment,)
        The second query for that similar subsequences in
        'sax_word_linearized_encoded' shall be found.
    :param start_linearized: np.array of shape (num_encoded_subsequences,)
        The index of the start (inclusive) of each encoded and linearized
        subsequence within its encoded and linearized SAX representation.
    :param end_linearized: np.array of shape (num_encoded_subsequences,)
        The index of the end (exclusive) of each encoded and linearized
        subsequence within its encoded and linearized SAX representation.
    :param normalize: bool
        True if the queries and the queried time series shall be z-normalized
        before computing distances.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between the
        two given queries and subsequences to check for their similarity.
    :return:
        A list that contains the starting indexes of the subsequences that
        belong to the motif of the two given queries.
    """

    fst_motif_idxs, snd_motif_idxs = [], []
    for idx, sax_word_encoded in enumerate(sax_word_split_encoded):
        sax_word_encoded = np.array(sax_word_encoded.iloc[:, 0]).astype(np.float64)

        fst_query = sax_word_encoded[start[idxs[0]]:end[idxs[0]]]
        fst_motif_idxs.extend(_find_motif(fst_query, sax_word_encoded, max_distance, normalize, p))

        snd_query = sax_word_encoded[start[idxs[1]]:end[idxs[1]]]
        snd_motif_idxs.extend(_find_motif(snd_query, sax_word_encoded, max_distance, normalize, p))

    # union, because all these subsequences are within 'max_distance' for at
    # least one of the two queries and for at least one symbolic position based
    # on the number of symbols per segment
    possible_motifs_idxs = list(set(fst_motif_idxs) | set(snd_motif_idxs))
    motif = []
    # find true motifs by considering 'max_distance' on full linearized SAX
    # words
    for idx in possible_motifs_idxs:
        possible_motif_linearized_encoded = sax_word_linearized_encoded.iloc[start_linearized[idx]:end_linearized[idx]]
        fst_dist = distance.minkowski(fst_query_linearized_encoded, possible_motif_linearized_encoded, p=p)
        snd_dist = distance.minkowski(snd_query_linearized_encoded, possible_motif_linearized_encoded, p=p)
        # true motif is within 'max_distance' of at least one of the two given
        # queries
        if fst_dist <= max_distance or snd_dist <= max_distance:
            motif.append(idx)

    return motif


def do_matrix_profile(df_norm, window_size, sax_variant, num_compare_segments,
                      max_distance, normalize=False, p=1.0, gap=1):
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

    df_sax_linearized_encoded, df_sax = _get_linearized_encoded_sax(df_norm, window_size, sax_variant)

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
        possible_motifs = _compute_possible_motifs(sax_word_split_encoded,
                                                   num_compare_segments,
                                                   float(max_distance), normalize, float(p))
        possible_motifs_idxs = list(possible_motifs.keys())
        ts_motifs = []
        # indexes that are already assigned to a motif
        assigned = set()
        sax_word_linearized_encoded = df_sax_linearized_encoded.iloc[:, idx]
        for idxs in possible_motifs_idxs:
            # motifs are disjoint
            # at least one of the two current subsequences already within
            # 'max_distance' of a previously encountered subsequence
            if idxs[0] in assigned or idxs[1] in assigned:
                continue

            # indexes for extraction of linearized encoded subsequences
            start_linearized = start * sax_variant.symbols_per_segment
            end_linearized = end * sax_variant.symbols_per_segment
            fst_query_linearized_encoded = sax_word_linearized_encoded.iloc[
                                           start_linearized[idxs[0]]:end_linearized[idxs[0]]
                                           ].reset_index(drop=True)
            snd_query_linearized_encoded = sax_word_linearized_encoded.iloc[
                                           start_linearized[idxs[1]]:end_linearized[idxs[1]]
                                           ].reset_index(drop=True)
            dist_queries = distance.minkowski(fst_query_linearized_encoded,
                                              snd_query_linearized_encoded, p=float(p))
            # check if the two potential motif queries are within
            # 'max_distance' of each other, and therefore form a motif
            if dist_queries > max_distance:
                continue

            # find all subsequences (respectively their indexes) that are
            # within 'max_distance' of at least one of the two queries
            # these subsequences belong to the motif of the two queries
            motif = _build_motif(sax_word_split_encoded, idxs, start, end,
                                 float(max_distance), sax_word_linearized_encoded,
                                 fst_query_linearized_encoded,
                                 snd_query_linearized_encoded,
                                 start_linearized, end_linearized, normalize, float(p))
            if motif:
                motif.sort()
                ts_motifs.append(motif)
                assigned.update(motif)

        motifs_lst.append(ts_motifs)

    # adjust start and end for indexes in original time series
    return motifs_lst, start * window_size, end * window_size
