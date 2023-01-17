from scipy.spatial import distance

from approximation.paa import PAA
from motif_discovery.matrix_profile import _encode_symbols
from discretization.sax.abstract_sax import linearize_sax_word
from utils.utils import constant_segmentation_overlapping


"""
References
----------
[1] Lonardi, J. L. E. K. S., & Patel, P. (2002, August). Finding motifs in time
series. In Proc. of the 2nd Workshop on Temporal Data Mining (pp. 53-68).
"""


def _hamming_distance(current_subsequence, subsequence):
    return sum(fst_num != snd_num
               for fst_num, snd_num in zip(current_subsequence, subsequence))


def _too_large_num_diff(current_subsequence, subsequence, num_diff_threshold):
    for fst_num, snd_num in zip(current_subsequence, subsequence):
        if abs(fst_num - snd_num) > num_diff_threshold:
            return True
    return False


def _find_best_motif(encoded_sax_subsequences, assigned, dist_threshold,
                     hamming_threshold, num_diff_threshold, start, p):
    best_count = 0
    best_motif = []
    initial = [(idx, subsequence)
               for idx, subsequence in enumerate(encoded_sax_subsequences)
               if start[idx] not in assigned]

    for current_subsequence in encoded_sax_subsequences:
        motif = []
        filtered = [(idx, subsequence) for idx, subsequence in initial
                    if distance.minkowski(current_subsequence, subsequence, p) <= dist_threshold]
        filtered = [(idx, subsequence) for idx, subsequence in filtered
                    if _hamming_distance(current_subsequence, subsequence) <= hamming_threshold]
        filtered = [(idx, subsequence) for idx, subsequence in filtered
                    if not _too_large_num_diff(current_subsequence, subsequence, num_diff_threshold)]

        motif.extend([start[idx] for idx, subs in filtered])

        if len(motif) > best_count:
            best_count = len(motif)
            best_motif = motif

    return best_motif


def do_brute_force(df_norm, window_size, sax_variant, num_compare_segments, p,
                   dist_threshold, hamming_threshold, num_diff_threshold):
    # TODO: factor out with Matrix Profile algorithm
    paa = PAA(window_size=window_size)
    df_paa = paa.transform(df_norm)

    df_sax = sax_variant.transform(df_paa=df_paa, df_norm=df_norm, window_size=window_size)
    if type(df_sax) is tuple:
        df_sax = df_sax[0]
    df_sax_linearized = linearize_sax_word(df_sax, sax_variant.symbols_per_segment)
    df_sax_linearized_encoded = _encode_symbols([[df_sax_linearized]], sax_variant)[0][0]

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
        assigned = set()
        while True:
            best_motif = _find_best_motif(encoded_sax_subsequences, assigned,
                                          dist_threshold, hamming_threshold,
                                          num_diff_threshold, start, p)
            if len(best_motif) <= 1:
                break
            ts_motifs.append(best_motif)
            assigned.update(best_motif)

        # adjust indexes for number of symbols per segment
        ts_motifs = [list(map(lambda idx: idx // sax_variant.symbols_per_segment, best_motif))
                     for best_motif in ts_motifs]
        motifs_lst.append(ts_motifs)

    # adjust for number of symbols per segment and discretization
    start = (start // sax_variant.symbols_per_segment) * window_size
    end = (end // sax_variant.symbols_per_segment) * window_size
    return motifs_lst, start, end
