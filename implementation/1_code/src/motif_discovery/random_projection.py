import random
import pandas as pd
import numpy as np

from approximation.paa import PAA
from discretization.sax.sax import SAX
from utils.utils import constant_segmentation_overlapping


def _get_sax_subsequences(df_norm, len_subsequence, window_size, alphabet_size, gap=1):
    """
    Extract subsequences of each of the given time series and transform each of
    them into its symbolic representation.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset whose subsequences shall be
        extracted and transformed into their symbolic representations.
    :param len_subsequence: int
        The length the extracted subsequences shall have.
    :param window_size: int
        The size of the window that is used to transform each subsequence into
        its PAA representation.
    :param alphabet_size: int
        The size of the alphabet that is used to transform the PAA
        representation of a subsequence into its symbolic representation.
    :param gap: int (default = 1)
        The gap between two consecutive subsequences within the corresponding
        time series when extracting them.
    :return:
        df_sax_lst: list of len(df_sax_lst) = num_ts
            Contains a dataframe of shape (num_subsequences, num_segments) for
            each time series with its corresponding subsequences.
            The subsequences are contained row-wise in their respective
            symbolic representation in each dataframe. The index of a
            subsequence within its dataframe corresponds to the extraction
            order of subsequences from the original time series. Hence, it
            serves as a pointer into 'start' and 'end' for the starting and
            ending index of the respective subsequence within its original time
            series.
        start: np.array of shape (num_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end: np.array of shape (num_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    """

    start, end = constant_segmentation_overlapping(df_norm.shape[0], len_subsequence, gap)
    num_ts = df_norm.shape[1]
    num_subsequences = len(start)

    df_subsequences_lst = []
    for i in range(num_ts):
        current_ts = df_norm.iloc[:, i]
        df_subsequences = pd.concat([current_ts[start[j]:end[j]].reset_index(drop=True)
                                     for j in range(num_subsequences)], axis=1)
        df_subsequences_lst.append(df_subsequences)

    # treat subsequences as whole time series
    paa = PAA(window_size=window_size)
    df_paa_lst = [paa.transform(df_subsequences_lst[i]) for i in range(num_ts)]

    # treat subsequences as whole time series
    sax = SAX(alphabet_size=alphabet_size)
    df_sax_lst = [sax.transform(df_paa_lst[i]) for i in range(num_ts)]
    # put subsequences of a time series row-wise in dataframe
    # then, index of subsequence corresponds to extraction order of
    # subsequences from original time series and serves as a pointer into
    # 'start' and 'end' for the starting and ending index of the respective
    # subsequence within the original time series
    df_sax_lst = [df_sax_lst[i].T.reset_index(drop=True) for i in range(num_ts)]

    return df_sax_lst, start, end


def _compute_collisions(df_sax_lst, num_projections, mask_size, seed=1):
    """
    Compute the number of collisions between each subsequence in its symbolic
    representation when projecting them into a lower-dimensional space at
    random.

    :param df_sax_lst: list of len(df_sax_lst) = num_ts
        Contains a dataframe of shape (num_subsequences, num_segments) for each
        time series with its corresponding subsequences.
        The subsequences are contained row-wise in their respective
        symbolic representation in each dataframe. The index of a
        subsequence within its dataframe corresponds to the extraction
        order of subsequences from the original time series.
    :param num_projections: int
        The number of consecutive random projections that shall be done for
        each given set of subsequences (i.e. dataframe).
    :param mask_size: int
        The dimension of the lower-dimensional space, the symbolic
        representations of the given subsequences shall be projected in.
    :param seed: int (default = 1)
        The seed that shall be used as a starting point to randomly choose the
        dimensions of the symbolic representations of the subsequences that
        shall be compared in the resulting lower-dimensional space.
    :return:
        collisions_lst: list of len(collisions_lst) = num_ts
            Contains a dictionary for each original time series. The keys are
            tuples containing the indexes of two subsequences. The values are
            the number of collisions of the symbolic representations
            corresponding to the subsequences encoded in the key that belongs
            to the respective value.
    """

    num_ts = len(df_sax_lst)
    collisions_lst = []
    for i in range(num_ts):
        current_subsequences = df_sax_lst[i]
        collisions = {}
        for it in range(num_projections):
            random.seed(seed + i + it)
            cols_mask = sorted(random.sample(range(current_subsequences.shape[1]), mask_size))
            df_sax_masked = pd.DataFrame(current_subsequences.iloc[:, cols_mask])

            # only compare current row with rows of higher index to avoid
            # duplicate comparisons
            start_row = 1
            # last row does not need any comparison
            for j in range(df_sax_masked.shape[0] - 1):
                compare_cols = df_sax_masked.iloc[start_row:]
                df_compare = compare_cols == df_sax_masked.iloc[j]
                # all symbols of the projected subsequence need to match
                df_match = compare_cols[df_compare.all(axis=1)]
                # index of subsequence in matching dataframe corresponds to
                # extraction order of subsequences from original time series
                # and serves as a pointer into 'start' and 'end' for the
                # starting and ending index of the respective subsequence
                # within the original time series
                for k in df_match.index.values:
                    collisions[(j, k)] = collisions.get((j, k), 0) + 1
                start_row += 1
        collisions_lst.append(collisions)

    return collisions_lst


def _filter_eucl_dist(df_norm_ts, start, end, promising_motifs_mindist, fst_matching_subsequence, snd_matching_subsequence, radius):
    motifs = []
    # check euclidean distance between the original subsequences
    # based on the promising motifs that fulfill MINDIST
    for idx in promising_motifs_mindist:
        fst_eucl_dist = np.linalg.norm(fst_matching_subsequence - df_norm_ts.iloc[start[idx]:end[idx]].reset_index(drop=True))
        snd_eucl_dist = np.linalg.norm(snd_matching_subsequence - df_norm_ts.iloc[start[idx]:end[idx]].reset_index(drop=True))
        if fst_eucl_dist <= radius or snd_eucl_dist <= radius:
            # current promising motif is actually within 'radius'
            # of at least one of the two current matching
            # subsequences
            motifs.append(idx)

    return motifs


def _filter_mindist(df_sax_ts, alphabet_size, promising_motifs, idxs, len_subsequence, radius):
    promising_motifs_sax_reprs = [(idx, df_sax_ts.iloc[idx]) for idx in promising_motifs]
    fst_matching_subsequence_sax_repr = df_sax_ts.iloc[idxs[0]]
    snd_matching_subsequence_sax_repr = df_sax_ts.iloc[idxs[1]]
    sax = SAX(alphabet_size=alphabet_size)

    # filter promising motifs based on MINDIST
    promising_motifs_mindist = []
    for idx, sax_repr in promising_motifs_sax_reprs:
        fst_mindist = sax.distance(pd.concat([sax_repr, fst_matching_subsequence_sax_repr], axis=1), len_subsequence).iloc[0, 1]
        snd_mindist = sax.distance(pd.concat([sax_repr, snd_matching_subsequence_sax_repr], axis=1), len_subsequence).iloc[0, 1]
        # if both is not fullfilled, then the euclidean distance
        # between the original subsequences is not within 'radius'
        # of both of the two current matching subsequences due to
        # the lower bounding property of MINDIST
        if fst_mindist <= radius or snd_mindist <= radius:
            promising_motifs_mindist.append(idx)

    return promising_motifs_mindist


def _get_motifs(df_norm, collisions_lst, radius, min_collisions, start, end, df_sax_lst, alphabet_size, len_subsequence):
    """
    Compute motifs based on the number of collisions between two symbolic
    representations corresponding to two subsequences.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset in which motifs shall be found based
        on subsequences and their symbolic representations.
    :param collisions_lst: list of len(collistions_lst) = num_ts
        The mapping between the subsequences of the original time series and
        the number of collisions between each two.
        Contains a dictionary for each original time series. The keys are
        tuples containing the indexes of two subsequences. The values are the
        number of collisions of the symbolic representations corresponding to
        the subsequences encoded in the key that belongs to the respective
        value.
    :param radius: float
        The maximum Euclidean distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :param min_collisions: int
        The minimum number of collisions between two subsequences based on
        their respective projected symbolic representations such that they are
        considered potential motifs that will be considered for finding motifs.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within its
        original time series.
    :param end: np.array of shape (num_subsequences,)
        The index of the end (exclusive) of each subsequence within its
        original time series.
    :param df_sax_lst: list of len(df_sax_lst) = num_ts
        Contains a dataframe of shape (num_subsequences, num_segments) for each
        time series with its corresponding subsequences.
        The subsequences are contained row-wise in their respective symbolic
        representation in each dataframe. The index of a subsequence within its
        dataframe corresponds to the extraction order of subsequences from the
        original time series. Hence, it serves as a pointer into 'start' and
        'end' for the starting and ending index of the respective subsequence
        within its original time series.
    :param alphabet_size: int
        The size of the alphabet that is used to compute the 'MINDIST' (SAX
        distance) between two subsequences in its symbolic representation.
    :param len_subsequence: int
        The length of the extracted subsequences from the original time series.
    :return:
        motifs_lst: list of len(motifs_lst) = num_ts
            Contains a two-dimensional list for each time series.
            In this two-dimensional list, each sub-list corresponds to a motif
            pattern. Hence, each of these sub-lists contains the indexes of the
            start and end of the respective motifs belonging to the respective
            motif pattern in the respective original time series.
            With such an index of a motif, the corresponding subsequence in the
            original time series can be queried with the help of 'start' and
            'end'.
    """

    num_ts = df_norm.shape[1]
    promising_collisions_lst = [{idxs: count for idxs, count in collisions_lst[i].items()
                                 if count >= min_collisions}
                                for i in range(num_ts)]
    # collisions ordered by count
    promising_collisions_lst = [dict(sorted(promising_collisions_lst[i].items(),
                                            key=lambda tup: tup[1], reverse=True))
                                for i in range(num_ts)]

    motifs_lst = []
    for i in range(num_ts):
        current_collisions_idxs = list(promising_collisions_lst[i].keys())
        ts_motifs = []
        for idxs in current_collisions_idxs:
            # delete current indexes to avoid exploring it again
            current_collisions_idxs[:] = [idxs_ for idxs_ in current_collisions_idxs if idxs_ != idxs]

            # indexes are pointers into 'start' and 'end' for the starting and
            # ending index of the respective subsequence within the original
            # time series
            fst_matching_subsequence = df_norm.iloc[start[idxs[0]]:end[idxs[0]], i].reset_index(drop=True)
            snd_matching_subsequence = df_norm.iloc[start[idxs[1]]:end[idxs[1]], i].reset_index(drop=True)
            eucl_dist_matching = np.linalg.norm(fst_matching_subsequence - snd_matching_subsequence)

            # contains all subsequences within 'radius' of at least one of the
            # two current matching subsequences
            motif = []
            if eucl_dist_matching <= radius:
                # the two current matching subsequences are a motif
                # start exploring further similar subsequences within 'radius'
                # of at least one of the two current matching subsequences
                motif.extend([idxs[0], idxs[1]])

                # all indexes i that fulfill (i, idxs[0]) and (idxs[0], i)
                fst_promising_motifs = [idxs_[0] for idxs_ in current_collisions_idxs if idxs_[1] == idxs[0]]\
                                       + [idxs_[1] for idxs_ in current_collisions_idxs if idxs_[0] == idxs[0]]
                # all indexes i that fulfill (i, idxs[1]) and (idxs[1], i)
                snd_promising_motifs = [idxs_[0] for idxs_ in current_collisions_idxs if idxs_[1] == idxs[1]]\
                                       + [idxs_[1] for idxs_ in current_collisions_idxs if idxs_[0] == idxs[1]]

                # promising motifs need to be similar to both of the two
                # current matching subsequences (i.e. high count for both)
                promising_motifs = list(set(fst_promising_motifs).intersection(snd_promising_motifs))
                promising_motifs_mindist = _filter_mindist(df_sax_lst[i], alphabet_size, promising_motifs, idxs, len_subsequence, radius)
                motifs = _filter_eucl_dist(df_norm.iloc[:, i], start, end, promising_motifs_mindist, fst_matching_subsequence, snd_matching_subsequence, radius)
                motif.extend(motifs)
                if motif:
                    motif.sort()
                    ts_motifs.append(motif)
                # motifs are disjoint
                current_collisions_idxs[:] = [idxs_ for idxs_ in current_collisions_idxs
                                              if idxs_[0] not in motif and idxs_[1] not in motif]

        motifs_lst.append(ts_motifs)

    return motifs_lst


def do_random_projection(df_norm, len_subsequence, window_size, alphabet_size, num_projections, mask_size, radius, min_collisions, gap=1, seed=1):
    df_sax_lst, start, end = _get_sax_subsequences(df_norm, len_subsequence, window_size, alphabet_size, gap)
    collisions_lst = _compute_collisions(df_sax_lst, num_projections, mask_size, seed)
    motifs_lst = _get_motifs(df_norm, collisions_lst, radius, min_collisions, start, end, df_sax_lst, alphabet_size, len_subsequence)
    return motifs_lst, start, end


def remove_trivial_motifs(motifs_per_subsequence_lst):
    """
    Remove trivial motifs for each subsequence.
    Given a subsequence 'S' with index 'a'. A trivial motif of 'S' is a motif
    of 'S' with index 'k' and all subsequences with indexes between 'a' and 'k'
    are motifs of 'S' as well.
    Intuitively, a motif that is near 'S' is not a trivial motif of 'S' if and
    only if there is at least one subsequence between them that is not a motif
    of 'S'.

    :param motifs_per_subsequence_lst:
        list of len(motifs_per_subsequence_lst) = num_ts
            It contains a dictionary for each time series. Each dictionary has
            the indexes of the subsequences of the respective time series that
            are contained in a motif pattern as keys. The value of a key is a
            list that contains the indexes of all subsequences that are motifs
            compared to the subsequence corresponding to the key.
            These lists may contain indexes of trivial motifs.
    :return:
        list of len(removed_lst) = num_ts
            It has the same structure as the input list
            'motifs_per_subsequence_lst', but with the indexes of trivial
            motifs removed.
    """

    removed_lst = []
    for ts_motifs in motifs_per_subsequence_lst:
        removed = {}
        for subsequence_idx, motif_lst in list(ts_motifs.items()):
            # start searching trivial matches of current subsequence and remove
            # them
            i, j = subsequence_idx-1, subsequence_idx+1
            while i in motif_lst:
                motif_lst.remove(i)
                i -= 1
            while j in motif_lst:
                motif_lst.remove(j)
                j += 1
            if motif_lst:
                removed[subsequence_idx] = motif_lst
        removed_lst.append(removed)

    return removed_lst


def get_motifs_per_subsequence(motifs_lst):
    """
    Compute all motifs that belong to a subsequence.

    :param motifs_lst: list of len(motifs_lst) = num_ts
        Contains a two-dimensional list for each time series.
        In this two-dimensional list, each sub-list corresponds to a motif
        pattern. Hence, each of these sub-lists contains the indexes of the
        start and end of the respective motifs belonging to the respective
        motif pattern in the respective original time series.
    :return:
        list of len(motifs_per_subsequence_lst) = num_ts
            It contains a dictionary for each time series. Each dictionary has
            the indexes of the subsequences of the respective time series that
            are contained in a motif pattern as keys. The value of a key is a
            list that contains the indexes of all subsequences that are motifs
            compared to the subsequence corresponding to the key.
    """

    motifs_per_subsequence_lst = []
    for ts_motifs in motifs_lst:
        motifs_per_subsequence = {}
        for motif_pattern in ts_motifs:
            for subsequence_idx in motif_pattern:
                motifs_per_subsequence[subsequence_idx] = [subsequence_idx_ for subsequence_idx_
                                                           in motif_pattern if subsequence_idx_ != subsequence_idx]
        # sort in an ascending order by the indexes of the subsequences
        motifs_per_subsequence_lst.append(dict(sorted(motifs_per_subsequence.items(),
                                                      key=lambda tup: tup[0])))

    return motifs_per_subsequence_lst
