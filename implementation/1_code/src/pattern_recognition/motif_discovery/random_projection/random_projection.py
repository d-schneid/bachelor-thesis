import random
import pandas as pd
import numpy as np

from approximation.paa import PAA
from discretization.sax.sax import SAX
from discretization.sax.one_d_sax import OneDSAX
from utils import constant_segmentation_overlapping, constant_segmentation
from discretization.sax.abstract_sax import linearize_sax_word
from pattern_recognition.motif_discovery.utils import _remove_trivial


"""
The functions below constitute the 'Random Projection' algorithm for motif
discovery in time series based on the extraction of subsequences and their
transformation to symbolic representations with SAX variants.

References
----------
[1] Chiu, Bill, Eamonn Keogh, and Stefano Lonardi. "Probabilistic discovery of
time series motifs." Proceedings of the ninth ACM SIGKDD international
conference on Knowledge discovery and data mining. 2003. (expanded version)
"""


def _get_sax_subsequences(df_norm, len_subsequence, window_size, sax_variant, gap=1):
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
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the PAA representation
        of a subsequence into its symbolic representation.
    :param gap: int (default = 1)
        The gap between two consecutive subsequences within the corresponding
        time series when extracting them.
    :return:
        df_sax_lst: list of len(df_sax_lst) = num_ts
            Contains a dataframe of shape
            (num_subsequences, num_segments * symbols_per_segment) for each
            time series with its corresponding subsequences.
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

    # treat subsequences as usual time series
    paa = PAA(window_size=window_size)
    df_paa_lst = [paa.transform(df_subsequences) for df_subsequences in df_subsequences_lst]

    # needed for the discretization approaches that compute individual
    # breakpoints for each time series (Persist, aSAX)
    df_breakpoints = sax_variant.compute_breakpoints(df_norm)
    df_breakpoints_lst = None
    if df_breakpoints is not None:
        # each subsequence of a time series shall be discretized based on the
        # same breakpoints
        df_breakpoints_lst = [pd.concat([df_breakpoints.iloc[:, i]] * num_subsequences, ignore_index=True, axis=1)
                              for i in range(df_breakpoints.shape[1])]

    # treat subsequences as usual time series
    df_sax_lst = []
    for i in range(num_ts):
        df_curr_breakpts = df_breakpoints_lst[i] if df_breakpoints_lst is not None else None
        df_sax = sax_variant.transform_to_symbolic_repr_only(df_paa=df_paa_lst[i], df_norm=df_subsequences_lst[i],
                                                             window_size=window_size, df_breakpoints=df_curr_breakpts)
        df_sax_lst.append(df_sax)

    df_sax_lst = [linearize_sax_word(df_sax, sax_variant.symbols_per_segment)
                  for df_sax in df_sax_lst]
    # put subsequences of a time series row-wise in dataframe
    # then, index of subsequence corresponds to extraction order of
    # subsequences from original time series and serves as a pointer into
    # 'start' and 'end' for the starting and ending index of the respective
    # subsequence within the original time series
    df_sax_lst = [df_sax.T.reset_index(drop=True) for df_sax in df_sax_lst]

    return df_sax_lst, start, end


def _compute_collisions(df_sax_lst, sax_variant, num_projections, mask_size, seed=1):
    """
    Compute the number of collisions between each subsequence in its symbolic
    representation when projecting them into a lower-dimensional space at
    random.

    :param df_sax_lst: list of len(df_sax_lst) = num_ts
        Contains a dataframe of shape
        (num_subsequences, num_segments * symbols_per_segment) for each time
        series with its corresponding subsequences.
        The subsequences are contained row-wise in their respective
        symbolic representation in each dataframe. The index of a
        subsequence within its dataframe corresponds to the extraction
        order of subsequences from the original time series.
    :param sax_variant: AbstractSAX
        The SAX variant that was used to transform the subsequences into their
        symbolic representations.
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
            num_cols_adjusted = current_subsequences.shape[1] // sax_variant.symbols_per_segment
            # randomly select segments that shall be used for projection
            cols_mask = sorted(random.sample(range(num_cols_adjusted), mask_size))
            cols_mask_adjusted = []
            for col_num in cols_mask:
                # first symbol of a segment
                col_num *= sax_variant.symbols_per_segment
                # catch all symbols of a segment
                cols_mask_adjusted.extend(list(range(col_num, col_num + sax_variant.symbols_per_segment)))
            df_sax_masked = pd.DataFrame(current_subsequences.iloc[:, cols_mask_adjusted])

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


def _filter_eucl_dist(df_norm_ts, start, end, promising_motifs, fst_matching_subsequence, snd_matching_subsequence, radius):
    """
    Compute the Euclidean distance between the two subsequences given by
    'fst_matching_subsequence' and 'snd_matching_subsequence' and all other
    subsequences given by 'promising_motifs'.
    This filters out the subsequences their Euclidean distance compared to the
    two given subsequences is greater than 'radius'. The remaining subsequences
    are said to be within 2 * 'radius' of the two given subsequences and are
    declared as motifs of these two.

    :param df_norm_ts: pd.Series of shape (ts_size,)
        The normalized time series whose subsequences shall be used to compute
        the Euclidean distance between them and the two given subsequences.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within its
        original time series.
    :param end: np.array of shape (num_subsequences,)
        The index of the end (exclusive) of each subsequence within its
        original time series.
    :param promising_motifs: list
        Contains the indexes of all subsequences based on the extraction order
        from the original time series that have a 'MINDIST' based on their
        symbolic representations that is smaller or equal than 'radius' with at
        least one of 'fst_matching_subsequence' and 'snd_matching_subsequence'.
    :param fst_matching_subsequence: pd.Series of shape (len_subsequences,)
        The first subsequence for which the motifs within 'radius' based on
        the Euclidean distance shall be found.
    :param snd_matching_subsequence: pd.Series of shape (len_subsequences,)
        The second subsequence for which the motifs within 'radius' based on
        the Euclidean distance shall be found.
    :param radius: float
        The maximum Euclidean distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :return:
        A list that contains the indexes of all subsequences based on the
        extraction order from the original time series that have a Euclidean
        distance that is smaller or equal than 'radius' with at least one of
        the two given subsequences of 'fst_matching_subsequence' and
        'snd_matching_subsequence'.
        These subsequences are declared as motifs of 'fst_matching_subsequence'
        and 'snd_matching_subsequence'.
    """

    motifs = []
    # check euclidean distance between the original subsequences
    # based on the promising motifs that fulfill MINDIST
    for idx in promising_motifs:
        fst_eucl_dist = np.linalg.norm(fst_matching_subsequence - df_norm_ts.iloc[start[idx]:end[idx]].reset_index(drop=True))
        snd_eucl_dist = np.linalg.norm(snd_matching_subsequence - df_norm_ts.iloc[start[idx]:end[idx]].reset_index(drop=True))
        if fst_eucl_dist <= radius or snd_eucl_dist <= radius:
            # current promising motif is actually within 'radius'
            # of at least one of the two current matching
            # subsequences
            motifs.append(idx)

    return motifs


def _filter_mindist(df_sax_ts, alphabet_size, promising_motifs, idxs, len_subsequence, radius):
    """
    Compute the 'MINDIST' (SAX distance) between the symbolic representations
    of the two subsequences given by 'idxs' and all other symbolic
    representations of subsequences given by 'promising_motifs'.
    This filters out the subsequences their Euclidean distance compared to the
    two subsequences given by 'idxs' is greater than 'radius'. It is based on
    the fact that the 'MINDIST' between the symbolic representations of two
    subsequences (or time series in general) lower bounds the Euclidean
    distance between these two subsequences (or time series in general).
    Therefore, if the 'MINDIST' is greater than 'radius' it automatically
    implies that the Euclidean distance is greater than 'radius'.

    :param df_sax_ts: dataframe of shape
        (num_subsequences, num_segments * symbols_per_segment)
        Contains the subsequences for a time series.
        The subsequences are contained row-wise in their respective symbolic
        representation. The index of a subsequence within the dataframe
        corresponds to the extraction order of subsequences from the original
        time series. Hence, it serves as a pointer into 'start' and 'end' for
        the starting and ending index of the respective subsequence within its
        original time series.
    :param alphabet_size: int
        The size of the alphabet that was used to compute the symbolic
        representations of the subsequences for which the 'MINDIST' shall be
        computed.
    :param promising_motifs: list of len(promising_motifs) = num_promising_motifs
        Contains the indexes of the subsequences based on the extraction order
        from the original time series that meet the minimum collision count
        threshold with both of the two given subsequences with indexes 'idxs'.
    :param idxs: tuple of shape (int, int)
        The indexes of the two subsequences based on the extraction order from
        the original time series for which the 'MINDIST' shall be computed on
        the symbolic representations with the given subsequences based on
        'promising_motifs'.
    :param len_subsequence: int
        The length of the subsequences for which the 'MINDIST' shall be
        computed.
    :param radius: float
        The maximum Euclidean distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :return:
        A list that contains the indexes of all subsequences based on the
        extraction order from the original time series that have a 'MINDIST'
        based on their symbolic representations that is smaller or equal than
        'radius' with at least one of the two given subsequences with indexes
        'idxs'.
    """

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


def _get_motifs(df_norm, collisions_lst, radius, min_collisions, start, end,
                df_sax_lst, sax_variant, len_subsequence, ignore_trivial,
                exclusion_zone):
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
        Contains a dataframe of shape
        (num_subsequences, num_segments * symbols_per_segment) for each time
        series with its corresponding subsequences.
        The subsequences are contained row-wise in their respective symbolic
        representation in each dataframe. The index of a subsequence within its
        dataframe corresponds to the extraction order of subsequences from the
        original time series. Hence, it serves as a pointer into 'start' and
        'end' for the starting and ending index of the respective subsequence
        within its original time series.
    :param sax_variant: AbstractSAX
        The SAX variant that was used to transform the subsequences into their
        symbolic representations and whose alphabet size is used to compute the
        'MINDIST' (SAX distance) between two subsequences in their symbolic
        representations.
        Note: The lower bounding property of the 'MINDIST' compared to the
        Euclidean distance is not fulfilled for the 1d-SAX. Therefore, for the
        1d-SAX the Euclidean distance between the original values of two
        subsequences is directly computed without any pre-filtering based on
        their 'MINDIST'.
    :param len_subsequence: int
        The length of the extracted subsequences from the original time series.
    :param ignore_trivial: bool
        If True: Trivial subsequence matches around 'exclusion_zone' (on the
        left and right side) are ignored for motif discovery.
    :param exclusion_zone: int
        The number of indexes on the left and right side of a match that shall
        be excluded from motif discovery.
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

    promising_collisions_lst = [{idxs: count for idxs, count in collisions.items()
                                 if count >= min_collisions}
                                for collisions in collisions_lst]
    # collisions ordered by count
    promising_collisions_lst = [dict(sorted(promising_collisions.items(),
                                            key=lambda tup: tup[1], reverse=True))
                                for promising_collisions in promising_collisions_lst]

    num_ts = df_norm.shape[1]
    motifs_lst = []
    for i in range(num_ts):
        current_collisions_idxs = list(promising_collisions_lst[i].keys())
        ts_motifs = []
        # indexes that are already assigned to a motif or declared as trivial
        excluded = set()
        for idxs in current_collisions_idxs:
            # motifs are disjoint
            # at least one of the two current subsequences already within
            # 'radius' of a previously encountered subsequence
            # or declared as trivial
            if idxs[0] in excluded or idxs[1] in excluded:
                continue

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
                promising_motifs = [idx for idx in promising_motifs if idx not in excluded]
                # lower bounding property of 'MINDIST' compared to Euclidean
                # distance is not fulfilled for 1d-SAX
                if not isinstance(sax_variant, OneDSAX):
                    promising_motifs = _filter_mindist(df_sax_lst[i], sax_variant.alphabet_size, promising_motifs, idxs, len_subsequence, radius)
                motifs = _filter_eucl_dist(df_norm.iloc[:, i], start, end, promising_motifs, fst_matching_subsequence, snd_matching_subsequence, radius)
                motif.extend(motifs)
                motif.sort()
                if ignore_trivial:
                    motif, trivial = _remove_trivial(motif, exclusion_zone)
                    # exclude subsequences in exclusion zones of subsequences
                    # in motif from motif discovery
                    excluded.update(trivial)
                if motif:
                    ts_motifs.append(motif)
                    excluded.update(motif)

        motifs_lst.append(ts_motifs)

    return motifs_lst


def do_random_projection(df_norm, len_subsequence, window_size, sax_variant,
                         num_projections, mask_size, radius, min_collisions,
                         ignore_trivial=True, exclusion_zone=None, gap=1, seed=1):
    """
    Execute the Random Projection algorithm for motif discovery in a time
    series based on its subsequences and their symbolic representations.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset for which motifs shall be found
        based on the extraction of subsequences and their symbolic
        representations.
    :param len_subsequence: int
        The length the extracted subsequences shall have.
    :param window_size: int
        The size of the window that is used to transform each subsequence into
        its PAA representation.
    :param sax_variant: AbstractSAX
        The SAX variant that shall be used to transform the PAA representation
        of a subsequence into its symbolic representation.
    :param num_projections: int
        The number of random projections of the symbolic representations of
        subsequences that shall be done to compare them in the resulting
        lower-dimensional spaces.
    :param mask_size: int
        The dimension of the lower-dimensional space, the symbolic
        representations of the given subsequences shall be projected in.
    :param radius: float
        The maximum Euclidean distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :param min_collisions: int
        The minimum number of collisions between two subsequences based on
        their respective projected symbolic representations such that they are
        considered potential motifs that will be considered for finding motifs.
    :param ignore_trivial: bool (default = True)
        If True: Trivial subsequence matches around 'exclusion_zone' (on the
        left and right side) of a subsequence are ignored for the motif they
        would be in, but also for further motifs.
    :param exclusion_zone: int (default = None)
        The number of indexes on the left and right side of a match that shall
        be excluded from motif discovery.
        If None, then it will be set to round('window_size' / 2).
        If 'ignore_trivial' is set to False, 'exclusion_zone' will be ignored.
    :param gap: int (default = 1)
        The gap between two consecutive subsequences within the corresponding
        time series when extracting them.
    :param seed: int (default = 1)
        The seed that shall be used as a starting point to randomly choose the
        dimensions of the symbolic representations of the subsequences that
        shall be compared in the resulting lower-dimensional space.
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
        start: np.array of shape (num_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end: np.array of shape (num_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    :raises:
        ValueError: If 'len_subsequence' > ts_size.
        ValueError: If 'window_size' > 'len_subsequence'.
        ValueError: If 'sax_variant' does not inherit from 'AbstractSAX'.
        ValueError: If 'mask_size' >= num_segments.
        ValueError: If 'min_collisions' > 'num_projections'.
    """

    if len_subsequence > df_norm.shape[0]:
        raise ValueError("The length of a subsequence needs to be smaller or "
                         "equal to the length of the time series the "
                         "subsequence shall be extracted.")
    if window_size > len_subsequence:
        raise ValueError("The size of the window with which the "
                         "subsequences are segmented needs to be smaller or "
                         "equal to the length of the subsequences.")
    num_segments = constant_segmentation(len_subsequence, window_size)[2]
    if mask_size >= num_segments:
        raise ValueError("The symbolic representations of the subsequences "
                         "need to be projected in a strictly lower-"
                         "dimensional space.")
    if min_collisions > num_projections:
        raise ValueError("The number of collisions between two symbolic "
                         "representations is strictly upper bounded by the "
                         "number of comparisons between these two symbolic "
                         "representations in a lower-dimensional space.")

    if not exclusion_zone:
        exclusion_zone = round(window_size / 2)

    df_sax_lst, start, end = _get_sax_subsequences(df_norm, len_subsequence,
                                                   window_size, sax_variant, gap)
    collisions_lst = _compute_collisions(df_sax_lst, sax_variant,
                                         num_projections, mask_size, seed)
    motifs_lst = _get_motifs(df_norm, collisions_lst, radius, min_collisions,
                             start, end, df_sax_lst, sax_variant,
                             len_subsequence, ignore_trivial, exclusion_zone)

    return motifs_lst, start, end
