import stumpy
import numpy as np
from scipy.spatial import distance

from approximation.paa import PAA
from utils.utils import constant_segmentation_overlapping
from discretization.sax.abstract_sax import linearize_sax_word
from discretization.sax.one_d_sax import OneDSAX


def _map_symbols_to_numbers_2(df_sax_linearized, sax_variant):
    mapping = dict(zip(sax_variant.alphabet,
                       [i for i in range(sax_variant.alphabet_size)]))

    if isinstance(sax_variant, OneDSAX):
        if sax_variant.alphabet_size_slope > sax_variant.alphabet_size:
            mapping = dict(zip(sax_variant.alphabet_slope,
                               [i for i in
                                range(sax_variant.alphabet_size_slope)]))

    return df_sax_linearized.replace(to_replace=mapping)


def _map_symbols_to_numbers(splitted, sax_variant):
    mapping = dict(zip(sax_variant.alphabet,
                       [i for i in range(sax_variant.alphabet_size)]))

    if isinstance(sax_variant, OneDSAX):
        if sax_variant.alphabet_size_slope > sax_variant.alphabet_size:
            mapping = dict(zip(sax_variant.alphabet_slope,
                               [i for i in
                                range(sax_variant.alphabet_size_slope)]))

    mapped = []
    for ts in splitted:
        mapped.append([df.replace(to_replace=mapping) for df in ts])

    return mapped


def _compute_matrix_profile_dikt(mapped_ts, len_symbolic_subsequence, max_distance):
    mps = []
    for df_mapped in mapped_ts:
        ts = np.array(df_mapped.iloc[:, 0]).astype(np.float64)
        mp = stumpy.stump(ts, m=len_symbolic_subsequence, normalize=False, p=1.0)[:, :2]
        mps.append(mp)

    dikts = []
    for mp in mps:
        dikt = {}
        for idx, elem in enumerate(mp):
            poss_match = mp[elem[1]]
            # nearest neighbor of poss_match already found
            # elem and poss_match are not nearest neighbors
            # not all subsequences have nearest neighbors, because distance is not symmetric
            # avoid checking pairs that are not nearest neighbors
            # and avoid checking nearest neighbors from again from other direction
            if (poss_match[1], elem[1]) in dikt or (elem[1], poss_match[1]) in dikt:
                continue
            # possible match points back, so it is a match
            # meaning distance is symmetric
            # does not need to point back, because it could have another
            # subsequence to which its distance is smaller
            if poss_match[1] == idx:
                # check if two subsequences are within 'max_distance' of each other
                # distance is symmetric: elem[0] == poss_match[0]
                if elem[0] <= max_distance:
                    dikt[(poss_match[1], elem[1])] = elem[0]

        dikts.append(dikt)
    # if same key, then take this element with maximum distance
    # already high distance for one symbolic dimension
    merged_dikt = {k: max(d.get(k, 0.0) for d in dikts) for k in set().union(*dikts)}

    # matches with the smallest distance first
    # retrieve most probable motifs first
    # every idx is contained at leaste once
    return dict(sorted(merged_dikt.items(), key=lambda tup: tup[1]))


def _match(ts_mapped, idxs, start, end, max_distance, ts_sax_linearized, query_fst_linearized, query_snd_linearized, start_linearized, end_linearized):
    # all indexes over all symbolic positions for the given two queries
    lst_fst, lst_snd = [], []
    # iterate through different symbolic positions
    for idx, mapped in enumerate(ts_mapped):
        ts = np.array(mapped.iloc[:, 0]).astype(np.float64)

        query_fst = np.array(mapped.iloc[start[idxs[0]]:end[idxs[0]], 0]).astype(np.float64)
        # self-match as well
        motif_idxs_fst = stumpy.match(query_fst, ts, max_distance=float(max_distance), max_matches=None, normalize=False, p=1.0)[:, 1]
        lst_fst.extend(motif_idxs_fst)

        query_snd = np.array(mapped.iloc[start[idxs[1]]:end[idxs[1]], 0]).astype(np.float64)
        motif_idxs_snd = stumpy.match(query_snd, ts, max_distance=float(max_distance), max_matches=None, normalize=False, p=1.0)[:, 1]
        lst_snd.extend(motif_idxs_snd)

    # union indexes, because all these indexes are within 'max_distance' in at
    # least one query over all symbolic positions
    to_be_checked = list(set(lst_fst) | set(lst_snd))
    motif = []
    # adjust for considering 'max_distance' over all symbolic positions
    # compute distance on linearized SAX representations
    for idx in to_be_checked:
        symbolic_ts = ts_sax_linearized.iloc[start_linearized[idx]:end_linearized[idx]]
        dist_fst = distance.minkowski(query_fst_linearized, symbolic_ts, p=1.0)
        dist_snd = distance.minkowski(query_snd_linearized, symbolic_ts, p=1.0)
        # find motifs within max_distance of at least one of the two subsequences
        # will also find itself again (distance is zero)
        if dist_fst <= max_distance or dist_snd <= max_distance:
            motif.append(idx)

    return motif


def make(df_norm, sax_variant, len_symbolic_subsequence, window_size, max_distance, gap=1):
    paa = PAA(window_size=window_size)
    df_paa = paa.transform(df_norm)

    df_sax = sax_variant.transform(df_paa=df_paa, df_norm=df_norm, window_size=window_size)
    if type(df_sax) is tuple:
        df_sax = df_sax[0]
    df_sax_linearized = linearize_sax_word(df_sax, sax_variant.symbols_per_segment)
    # TODO: adjust _map_symbols function
    df_sax_linearized = _map_symbols_to_numbers_2(df_sax_linearized, sax_variant)

    # split if there are multiple symbols per segment
    # one list element for each symbol position
    splitted = []
    for col_name, sax_repr in df_sax.items():
        sax_repr = sax_repr.to_frame()
        splitted.append([sax_repr.applymap(lambda symbols: symbols[i]) for i in range(sax_variant.symbols_per_segment)])
    mapped = _map_symbols_to_numbers(splitted, sax_variant)

    # start and end for symbolic representation, respectively its encoded representation
    start, end = constant_segmentation_overlapping(df_sax.shape[0], len_symbolic_subsequence, gap)
    motifs_lst = []
    for idx, ts_mapped in enumerate(mapped):
        # find pairs of two very good matches, compute neareast neighbor
        # dikt already merged
        dikt_mp = _compute_matrix_profile_dikt(ts_mapped, len_symbolic_subsequence, max_distance=max_distance)
        current_distance_idxs = list(dikt_mp.keys())
        ts_motifs = []
        removed = set()
        # linearized version of SAX representation for whole words extraction of current time series
        ts_sax_linearized = df_sax_linearized.iloc[:, idx]
        for idxs in current_distance_idxs:
            # motifs are disjoint
            # current index is already contained in a motif
            # i.e. it is within max_distance of a previous index (i.e. the subsequence corresponding to it)
            if idxs[0] in removed or idxs[1] in removed:
                continue

            # start and end indexes for whole words, not splitted words
            start_linearized = start * sax_variant.symbols_per_segment
            end_linearized = end * sax_variant.symbols_per_segment
            # extract whole word queries, not splitted
            query_fst_linearized = ts_sax_linearized.iloc[start_linearized[idxs[0]]:end_linearized[idxs[0]]].reset_index(drop=True)
            query_snd_linearized = ts_sax_linearized.iloc[start_linearized[idxs[1]]:end_linearized[idxs[1]]].reset_index(drop=True)
            # check if two whole words build a motif (i.e. are within 'max_distance')
            dist_queries = distance.minkowski(query_fst_linearized, query_snd_linearized, p=1.0)
            # queries are not within 'max_distance' of each other
            # otherwise they form a motif
            if dist_queries > max_distance:
                continue

            # find all subsequences that are within 'max_distance' of at least one of the queries
            motif = _match(ts_mapped, idxs, start, end, max_distance, ts_sax_linearized, query_fst_linearized, query_snd_linearized, start_linearized, end_linearized)
            # motif contains indexes wrt to start of symbolic subsequence
            if motif:
                motif.sort()
                ts_motifs.append(motif)
                removed.update(motif)

        # motifs for each given time series
        motifs_lst.append(ts_motifs)

    # adjust start and end for indexes in original time series
    return motifs_lst, start * window_size, end * window_size
