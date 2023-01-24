import stumpy
from scipy.spatial import distance

from utils import constant_segmentation_overlapping
from pattern_recognition.motif_discovery.matrix_profile.shared import _find_nearest_neighs


"""
The functions below constitute an algorithm for motif discovery in time series
that is based on the Matrix Profile. The Matrix Profile is applied on the raw
values of given time series.
Moreover, the idea of finding similar subsequences in a bounded area around
two selected and promising subsequences from reference [1] is used to discover
motifs.

References
----------
[1] Chiu, Bill, Eamonn Keogh, and Stefano Lonardi. "Probabilistic discovery of
time series motifs." Proceedings of the ninth ACM SIGKDD international
conference on Knowledge discovery and data mining. 2003. (expanded version)
"""


def _compute_nearest_neighs(df_norm, len_subsequence, max_distance, normalize, p):
    """
    For each time series of the given time series dataset, find nearest
    neighbor pairs of its subsequences based on the computation of the Matrix
    Profile.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series for that Matrix Profiles are computed for finding
        pairs of nearest neighbors.
    :param len_subsequence: int
        The length of the extracted subsequences for that pairs of nearest
        neighbors shall be found.
    :param max_distance: float
        The maximum distance between two nearest neighbors sucht that they are
        declared a potential motif.
    :param normalize: bool
        True if subsequences shall be z-normalized before computing distances.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between the
        two subsequences to check for their similarity.
    :return: list of len(num_ts)
        Contains a dictionary for each given time series. Each dictionary
        contains tuples with the indexes of two nearest neighbors as keys and
        the corresponding distance as values. The subsequences corresponding to
        the contained indexes are considered as a potential motif.
    """

    matrix_profiles = []
    for col_name, col_ts in df_norm.items():
        matrix_profile = stumpy.stump(col_ts, m=len_subsequence, normalize=normalize, p=p)[:, :2]
        matrix_profiles.append(matrix_profile)

    nearest_neighs_lst = []
    for matrix_profile in matrix_profiles:
        nearest_neighs = _find_nearest_neighs(matrix_profile, max_distance)
        # sort nearest neighbors in ascending order wrt distance
        nearest_neighs_lst.append(dict(sorted(nearest_neighs.items(), key=lambda tup: tup[1])))

    return nearest_neighs_lst


def _build_motif(current_ts, fst_neigh, snd_neigh, max_distance, normalize, p):
    """
    Find all subsequences within the given time series that are within
    'max_distance' of at least one of the two given nearest neighbors that form
    a motif and include them into this motif.

    :param current_ts: pd.Series of shape (ts_size,)
        The time series to find motifs in.
    :param fst_neigh: pd.Series of shape (len_subsequence,)
        One of the two nearest neighbor subsequences that form a motif.
    :param snd_neigh: pd.Series of shape (len_subsequence,)
        The other of the two nearest neighbor subsequences that form a motif.
    :param max_distance: float
        The maximum distance between one of the two given subsequences and
        another subsequence from the given time series, such the other
        subsequence will be included into their motif.
    :param normalize: bool
        True if the subsequences shall be z-normalized before comparison.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between two
        subsequences to check for their similarity.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within the given
        time series.
    :param end: np.array of shape (num_subsequences,)
        The index of the end (exclusive) of each subsequence within the given
        time series.
    :return: list
        Contains the indexes of the start and end of the respective
        subsequences within the given time series that belong to this motif.
        With such an index, the corresponding subsequence in the original time
        series can be queried with the help of 'start' and 'end'.
    """

    fst_motif_idxs, snd_motif_idxs = [], []
    fst_motif_idxs.extend(stumpy.match(fst_neigh, current_ts, max_distance=max_distance,
                                       normalize=normalize, p=p)[:, 1])
    snd_motif_idxs.extend(stumpy.match(snd_neigh, current_ts, max_distance=max_distance,
                                       normalize=normalize, p=p)[:, 1])

    motif = list(set(fst_motif_idxs) | set(snd_motif_idxs))
    return motif


def _is_motif(current_ts, idxs, assigned, max_distance, p, start, end):
    """
    Check if the two given subsequences corresponding to 'idxs' form a motif.

    :param current_ts: pd.Series of shape (ts_size,)
        The time series to find motifs in.
    :param idxs: tuple
        Contains the indexes based on 'start' and 'end' of the two subsequences
        within the given time series that are declared as nearest neighbors.
    :param assigned: set
        The indexes of all subsequences within the givne time series that have
        already been assigned to a motif.
    :param max_distance: float
        The maximum distance between the two subsequences corresponding to
        'idxs', such that they are declared a motif.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between the
        two subsequences corresponding to 'idxs' to check for their similarity.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within the given
        time series.
    :param end: np.array of shape (num_subsequences,)
        The index of the end (exclusive) of each subsequence within the given
        time series.
    :return: tuple
        first entry: bool
            True: The two subsequences corresponding to 'idxs' form a motif.
            False: Otherwise.
        second entry (only if first entry is 'True'): pd.Series of shape (len_subsequence,)
            One of the two subsequences corresponding to 'idxs'.
        third entry (only if first entry is 'True): pd.Series of shape (len_subsequence,)
            The other of the two subsequences corresponding to 'idxs'.
    """

    # motifs are disjoint
    # at least one of the two current subsequences already within
    # 'max_distance' of a previously encountered subsequence
    if idxs[0] in assigned or idxs[1] in assigned:
        return False,

    fst_neigh = current_ts[start[idxs[0]]:end[idxs[0]]]
    snd_neigh = current_ts[start[idxs[1]]:end[idxs[1]]]
    dist_queries = distance.minkowski(fst_neigh, snd_neigh, p=float(p))

    # check if the two potential motif queries are within
    # 'max_distance' of each other, and therefore form a motif
    if dist_queries > max_distance:
        return False,

    return True, fst_neigh, snd_neigh


def _build_ts_motifs(current_ts, nearest_neighs, max_distance, normalize, p, start, end):
    """
    Compute all motifs of the given time series.

    :param current_ts: pd.Series of shape (ts_size,)
        The time series to find motifs in.
    :param nearest_neighs: dict
        Contains tuples with the indexes of two nearest subsequences as keys
        and the corresponding distance as values.
        The subsequences corresponding to the contained indexes are considered
        as a possible motif.
    :param max_distance: float
        The upper bound for the distance between a subsequence and a query,
        such that the subsequence is considered similar to the query.
    :param normalize: bool
        True if the subsequence and the query shall be z-normalized before
        comparison.
        False otherwise.
    :param p: float
        The p-norm to apply for computing the Minkowski distance between a
        subsequence and a query to check for their similarity.
    :param start: np.array of shape (num_subsequences,)
        The index of the start (inclusive) of each subsequence within its
        original time series.
    :param end:
        The index of the end (exclusive) of each subsequence within its
        original time series.
    :return: list
        Contains a list for each found motif in the given time series.
        Each of these sub-lists contains the indexes of the start and end of
        the respective subsequences within the given time series.
        With such an index, the corresponding subsequence in the original time
        series can be queried with the help of 'start' and 'end'.
    """

    nearest_neighs_idxs = list(nearest_neighs.keys())
    ts_motifs = []
    # indexes that are already assigned to a motif
    assigned = set()
    for idxs in nearest_neighs_idxs:
        new_motif = _is_motif(current_ts, idxs, assigned, max_distance, p, start, end)
        if not new_motif[0]:
            continue

        fst_neigh, snd_neigh = new_motif[1], new_motif[2]
        motif = _build_motif(current_ts, fst_neigh, snd_neigh, max_distance,
                             normalize, p)
        if motif:
            motif.sort()
            ts_motifs.append(motif)
            assigned.update(motif)

    return ts_motifs


def do_matrix_profile_raw(df_norm, len_subsequence, max_distance, normalize=False, p=1.0, gap=1):
    """
    Execute the algorithm based on the computation of the Matrix Profile for
    motif discovery in a time series based on its subsequences.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The normalized time series dataset for which motifs shall be found
        based on the extraction of subsequences.
    :param len_subsequence: int
        The length of the extracted subsequences of a time series.
    :param max_distance: float
        The maximum distance between two subsequences, such that they
        are considered similar enough to be declared as motifs.
    :param normalize: bool (default = False)
        True if the distances between two subsequences shall be computed on
        z-normalized versions of them.
        False otherwise.
    :param p: float (default = 1.0)
        The p-norm to apply for computing the Minkowski distance between two
        subsequences to check for their similarity.
    :param gap: int (default = 1)
        The gap between two consecutive subsequences within the corresponding
        time series when extracting them.
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
        start: np.array of shape (num_subsequences,)
            The index of the start (inclusive) of each subsequence within its
            original time series.
        end: np.array of shape (num_subsequences,)
            The index of the end (exclusive) of each subsequence within its
            original time series.
    :raises:
        ValueError: If 'len_subsequence' > ts_size.
    """

    if len_subsequence > df_norm.shape[0]:
        raise ValueError("The length of the subsequences cannot be larger "
                         "than the length of the time series.")

    nearest_neighs_lst = _compute_nearest_neighs(df_norm, len_subsequence, max_distance, normalize, p)
    start, end = constant_segmentation_overlapping(df_norm.shape[0], len_subsequence, gap)
    motifs_lst = []
    for idx, nearest_neighs in enumerate(nearest_neighs_lst):
        current_ts = df_norm.iloc[:, idx]
        # build up motifs based nearest neighbors
        ts_motifs = _build_ts_motifs(current_ts, nearest_neighs, max_distance, normalize, p, start, end)
        motifs_lst.append(ts_motifs)

    return motifs_lst, start, end
