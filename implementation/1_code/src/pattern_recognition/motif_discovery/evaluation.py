import numpy as np
from collections import Counter

from pattern_recognition.motif_discovery.brute_force.discretized import do_brute_force_discretized
from pattern_recognition.motif_discovery.brute_force.raw import do_brute_force_raw
from pattern_recognition.motif_discovery.random_projection.random_projection import do_random_projection
from pattern_recognition.motif_discovery.matrix_profile.discretized import do_matrix_profile_discretized
from pattern_recognition.motif_discovery.matrix_profile.raw import do_matrix_profile_raw


def compute_most_common(lst):
    arr = np.array(lst)
    unique_values, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    max_count_indices = np.where(counts == max_count)[0]
    non_negative_indices = max_count_indices[unique_values[max_count_indices] >= 0]
    if len(non_negative_indices) > 0:
        most_common = np.max(unique_values[non_negative_indices])
    else:
        most_common = unique_values[np.argmax(counts)]
    return most_common


# in each bin the motif-id is assigned to the motifs within the bin
# for each bin the motif-id that most often occurs is assigned as the motif-id for the whole bin
def assign_motifs_to_ts(ts_motifs_lst, start, end, df_labels):
    ts_result = []
    for motif_bin in ts_motifs_lst:
        bin_result = []
        for motif_idx in motif_bin:
            motif_counts = df_labels.iloc[start[motif_idx]:end[motif_idx]]["pattern_main_id"].value_counts()
            predicted_motif = motif_counts.idxmax()
            bin_result.append(predicted_motif)
        motif_result = compute_most_common(bin_result)
        if motif_result == -1:
            print("-1")
        ts_result.append(motif_result)
    return ts_result


# for each bin count the total number of points in the time series that are assigned to the motif-id assigned to the bin --> true positives + false negatives
# for the motifs in each bin sum up the number of points that are assigned to the assigned motif-id assigned to the bin --> true positives
# divide the total number by the number across the motifs in the bin
# result: recall for each bin
def recall_for_ts(ts_motifs_lst, start, end, df_labels, assigned_motifs_lst):
    ts_bin_recall_lst = []
    for i, motif_bin in enumerate(ts_motifs_lst):
        assigned_motif = assigned_motifs_lst[i]
        total_assigned_motifs = df_labels["pattern_main_id"].value_counts()[assigned_motif]
        num_bin_motifs = 0
        for motif_idx in motif_bin:
            motif_labels = df_labels.iloc[start[motif_idx]:end[motif_idx]]["pattern_main_id"]
            num_bin_motifs += motif_labels.value_counts().get(assigned_motif, 0)
        bin_recall = num_bin_motifs / total_assigned_motifs
        ts_bin_recall_lst.append(bin_recall)
    return ts_bin_recall_lst


# for the motifs in each bin sum up the number of points that are assigned to the assigned motif-id assigned to the bin --> true positives
# for each bin divide this sum by the total number of points across the motifs in the bin --> true positives + false positives
# result: precision for each bin
def precision_for_ts(ts_motifs_lst, start, end, df_labels, assigned_motifs_lst, motif_length):
    ts_bin_precision_lst = []
    for i, motif_bin in enumerate(ts_motifs_lst):
        assigned_motif = assigned_motifs_lst[i]
        numerator = 0
        for motif_idx in motif_bin:
            motif_labels = df_labels.iloc[start[motif_idx]:end[motif_idx]]["pattern_main_id"]
            numerator += motif_labels.value_counts().get(assigned_motif, 0)
        ts_bin_precision_lst.append(numerator / (len(motif_bin) * motif_length))
    return ts_bin_precision_lst


def found_motifs_rate_for_ts(assigned_motifs_lst, df_labels, motif_length):
    motifs = df_labels.value_counts()
    if -1 in motifs.index:
        motifs = motifs.drop(-1, level=0)
    # motif needs to have at least 2 occurrences
    motifs = motifs[motifs >= 2 * motif_length]
    num_motifs_ts = motifs.size
    num_motifs_found = len(set(assigned_motifs_lst))
    return num_motifs_found / num_motifs_ts


def compute_mean_bins_per_motif(assigned_motifs_lst):
    count = Counter(assigned_motifs_lst)
    return sum(count.values()) / len(count)


def compute_noise_to_motif_ratio(assigned_motifs_lst):
    noise = len([assigned_motif for assigned_motif in assigned_motifs_lst if assigned_motif == -1])
    return noise / len(assigned_motifs_lst)


def clean(ts_motifs, assigned_motifs_lst):
    noise_indexes = [i for i, assigned_motif in enumerate(assigned_motifs_lst) if assigned_motif == -1]
    ts_motifs_clean = [ts_motif for i, ts_motif in enumerate(ts_motifs) if i not in noise_indexes]
    assigned_motifs_lst_clean = [assigned_motif for assigned_motif in assigned_motifs_lst if assigned_motif != -1]
    return ts_motifs_clean, assigned_motifs_lst_clean


def _evaluate(motifs_lst, start, end, df_labels):
    motif_length = end[0] - start[0]
    print(f"motif length: {motif_length}")
    ts_motifs = motifs_lst[0]
    assigned_motifs_lst = assign_motifs_to_ts(ts_motifs, start, end, df_labels)
    ts_motifs_clean, assigned_motifs_lst_clean = clean(ts_motifs, assigned_motifs_lst)

    ts_recall = recall_for_ts(ts_motifs_clean, start, end, df_labels, assigned_motifs_lst_clean)
    ts_precision = precision_for_ts(ts_motifs_clean, start, end, df_labels, assigned_motifs_lst_clean, motif_length)
    ts_found_motifs_rate = found_motifs_rate_for_ts(assigned_motifs_lst_clean, df_labels, motif_length)
    ts_mean_bins_per_motif = compute_mean_bins_per_motif(assigned_motifs_lst_clean)
    ts_noise_to_motif_ratio = compute_noise_to_motif_ratio(assigned_motifs_lst)
    return ts_recall, ts_precision, ts_found_motifs_rate, ts_mean_bins_per_motif, ts_noise_to_motif_ratio


# final SAX, aSAX, Persist
# alphabet_size = 19
#window_size_bf = 5
#num_compare_segments_bf = 24
#p_bf = 1.0
#dist_threshold_bf = 15
#num_diff_threshold_bf = 2
#hamming_threshold_bf = 15
#ignore_trivial_bf = True
#exclusion_zone_bf = 12


# final SAX, aSAX, Persist
# alphabet_size = 16
window_size_bf = 5
num_compare_segments_bf = 12
p_bf = 1.0
dist_threshold_bf = 10
num_diff_threshold_bf = 1
hamming_threshold_bf = 10
ignore_trivial_bf = True
exclusion_zone_bf = 10


def evaluate_brute_force(df_motifs, df_labels, sax_variant):
    motifs_lst, start, end = do_brute_force_discretized(df_motifs, window_size_bf, sax_variant, num_compare_segments_bf,
                                                        dist_threshold_bf, num_diff_threshold_bf, hamming_threshold_bf, p_bf,
                                                        ignore_trivial=ignore_trivial_bf, exclusion_zone=exclusion_zone_bf)
    return _evaluate(motifs_lst, start, end, df_labels)


# final 1d-SAX
# alphabet_size = 19
# alphabet_size_slope = 3
window_size_bf_od = 10
num_compare_segments_bf_od = 12
p_bf_od = 1.0
dist_threshold_bf_od = 16
num_diff_threshold_bf_od = 2
hamming_threshold_bf_od = 16
ignore_trivial_bf_od = True
exclusion_zone_bf_od = 10


# final 1d-SAX
# alphabet_size = 23
# alphabet_size_slope = 3
#window_size_bf_od = 10
#num_compare_segments_bf_od = 6
#p_bf_od = 1.0
#dist_threshold_bf_od = 7
#num_diff_threshold_bf_od = 2
#hamming_threshold_bf_od = 7
#ignore_trivial_bf_od = True
#exclusion_zone_bf_od = 5


def evaluate_brute_force_one_d_sax(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: window size {window_size_bf_od}")
    print(f"{sax_variant.name}: num compare segments {num_compare_segments_bf_od}")
    print(f"{sax_variant.name}: p {p_bf_od}")
    print(f"{sax_variant.name}: dist threshold {dist_threshold_bf_od}")
    print(f"{sax_variant.name}: num diff threshold {num_diff_threshold_bf_od}")
    print(f"{sax_variant.name}: hamming threshold {hamming_threshold_bf_od}")
    print(f"{sax_variant.name}: ignore trivial {ignore_trivial_bf_od}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_bf_od}")
    motifs_lst, start, end = do_brute_force_discretized(df_motifs, window_size_bf_od, sax_variant, num_compare_segments_bf_od,
                                                        dist_threshold_bf_od, num_diff_threshold_bf_od, hamming_threshold_bf_od, p_bf_od,
                                                        ignore_trivial=ignore_trivial_bf_od, exclusion_zone=exclusion_zone_bf_od)
    return _evaluate(motifs_lst, start, end, df_labels)


# final eSAX
# alphabet_size = 12
window_size_bf_e = 10
num_compare_segments_bf_e = 12
p_bf_e = 1.0
dist_threshold_bf_e = 25
num_diff_threshold_bf_e = 2
hamming_threshold_bf_e = 25
ignore_trivial_bf_e = True
exclusion_zone_bf_e = 10


# final eSAX
# alphabet_size = 14
#window_size_bf_e = 10
#num_compare_segments_bf_e = 6
#p_bf_e = 1.0
#dist_threshold_bf_e = 12
#num_diff_threshold_bf_e = 5
#hamming_threshold_bf_e = 12
#ignore_trivial_bf_e = True
#exclusion_zone_bf_e = 5


def evaluate_brute_force_e_sax(df_motifs, df_labels, sax_variant):
    motifs_lst, start, end = do_brute_force_discretized(df_motifs, window_size_bf_e, sax_variant, num_compare_segments_bf_e,
                                                        dist_threshold_bf_e, num_diff_threshold_bf_e, hamming_threshold_bf_e, p_bf_e,
                                                        ignore_trivial=ignore_trivial_bf_e, exclusion_zone=exclusion_zone_bf_e)
    return _evaluate(motifs_lst, start, end, df_labels)


# final raw
p_bf_raw = 1.0
dist_threshold_bf_raw = 15
num_diff_threshold_bf_raw = 0.5
len_subsequence_bf_raw = 120
ignore_trivial_bf_raw = True
exclusion_zone_bf_raw = 40


def evaluate_brute_force_raw(df_motifs, df_labels):
    motifs_lst, start, end = do_brute_force_raw(df_motifs, len_subsequence_bf_raw, dist_threshold_bf_raw,
                                                num_diff_threshold_bf_raw, p_bf_raw, ignore_trivial_bf_raw, exclusion_zone_bf_raw)
    return _evaluate(motifs_lst, start, end, df_labels)


# final SAX, aSAX, Persist
# alphabet_size = 26
len_subsequence_rp = 60
window_size_rp = 5
num_projections_rp = 12
mask_size_rp = 2
radius_rp = 1
min_collisions_rp = 0
ignore_trivial_rp = True
exclusion_zone_rp = 60


def evaluate_random_projection(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: len subsequence {len_subsequence_rp}")
    print(f"{sax_variant.name}: window size {window_size_rp}")
    print(f"{sax_variant.name}: num projections {num_projections_rp}")
    print(f"{sax_variant.name}: mask size {mask_size_rp}")
    print(f"{sax_variant.name}: radius {radius_rp}")
    print(f"{sax_variant.name}: min collisions {min_collisions_rp}")
    print(f"{sax_variant.name}: ignore trivial {ignore_trivial_rp}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_rp}")
    motifs_lst, start, end = do_random_projection(df_motifs, len_subsequence_rp, window_size_rp,
                                                  sax_variant, num_projections_rp, mask_size_rp,
                                                  radius_rp, min_collisions_rp, ignore_trivial_rp,
                                                  exclusion_zone_rp)
    return _evaluate(motifs_lst, start, end, df_labels)


len_subsequence_rp_od = 120
window_size_rp_od = 5
num_projections_rp_od = 18
mask_size_rp_od = 2
radius_rp_od = 2.2
min_collisions_rp_od = 4
ignore_trivial_rp_od = True
exclusion_zone_rp_od = 100


def evaluate_random_projection_one_d_sax(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: len subsequence {len_subsequence_rp_od}")
    print(f"{sax_variant.name}: window size {window_size_rp_od}")
    print(f"{sax_variant.name}: num projections {num_projections_rp_od}")
    print(f"{sax_variant.name}: mask size {mask_size_rp_od}")
    print(f"{sax_variant.name}: radius {radius_rp_od}")
    print(f"{sax_variant.name}: min collisions {min_collisions_rp_od}")
    print(f"{sax_variant.name}: ignore trivial {ignore_trivial_rp_od}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_rp_od}")
    motifs_lst, start, end = do_random_projection(df_motifs, len_subsequence_rp_od, window_size_rp_od,
                                                  sax_variant, num_projections_rp_od, mask_size_rp_od,
                                                  radius_rp_od, min_collisions_rp_od, ignore_trivial_rp_od,
                                                  exclusion_zone_rp_od)
    return _evaluate(motifs_lst, start, end, df_labels)


len_subsequence_rp_od2 = 120
window_size_rp_od2 = 10
num_projections_rp_od2 = 18
mask_size_rp_od2 = 2
radius_rp_od2 = 2.2
min_collisions_rp_od2 = 3
ignore_trivial_rp_od2 = True
exclusion_zone_rp_od2 = 100


def evaluate_random_projection_one_d_sax2(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}2: len subsequence {len_subsequence_rp_od2}")
    print(f"{sax_variant.name}2: window size {window_size_rp_od2}")
    print(f"{sax_variant.name}2: num projections {num_projections_rp_od2}")
    print(f"{sax_variant.name}2: mask size {mask_size_rp_od2}")
    print(f"{sax_variant.name}2: radius {radius_rp_od2}")
    print(f"{sax_variant.name}2: min collisions {min_collisions_rp_od2}")
    print(f"{sax_variant.name}2: ignore trivial {ignore_trivial_rp_od2}")
    print(f"{sax_variant.name}2: exclusion zone {exclusion_zone_rp_od2}")
    motifs_lst, start, end = do_random_projection(df_motifs, len_subsequence_rp_od2, window_size_rp_od2,
                                                  sax_variant, num_projections_rp_od2, mask_size_rp_od2,
                                                  radius_rp_od2, min_collisions_rp_od2, ignore_trivial_rp_od2,
                                                  exclusion_zone_rp_od2)
    return _evaluate(motifs_lst, start, end, df_labels)


len_subsequence_rp_e = 120
window_size_rp_e = 5
num_projections_rp_e = 32
mask_size_rp_e = 2
radius_rp_e = 2.2
min_collisions_rp_e = 4
ignore_trivial_rp_e = True
exclusion_zone_rp_e = 100


def evaluate_random_projection_e_sax(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: len subsequence {len_subsequence_rp_e}")
    print(f"{sax_variant.name}: window size {window_size_rp_e}")
    print(f"{sax_variant.name}: num projections {num_projections_rp_e}")
    print(f"{sax_variant.name}: mask size {mask_size_rp_e}")
    print(f"{sax_variant.name}: radius {radius_rp_e}")
    print(f"{sax_variant.name}: min collisions {min_collisions_rp_e}")
    print(f"{sax_variant.name}: ignore trivial {ignore_trivial_rp_e}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_rp_e}")
    motifs_lst, start, end = do_random_projection(df_motifs, len_subsequence_rp_e, window_size_rp_e,
                                                  sax_variant, num_projections_rp_e, mask_size_rp_e,
                                                  radius_rp_e, min_collisions_rp_e, ignore_trivial_rp_e,
                                                  exclusion_zone_rp_e)
    return _evaluate(motifs_lst, start, end, df_labels)


window_size_mp = 10
num_compare_segments_mp = 6
max_distance_mp = 7.0
exclusion_zone_mp = 12
p_mp = 1.0


def evaluate_matrix_profile(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: window size {window_size_mp}")
    print(f"{sax_variant.name}: num compare segments {num_compare_segments_mp}")
    print(f"{sax_variant.name}: max distance {max_distance_mp}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_mp}")
    print(f"{sax_variant.name}: p {p_mp}")
    motifs_lst, start, end = do_matrix_profile_discretized(df_motifs, window_size_mp, sax_variant,
                                                           num_compare_segments_mp, max_distance_mp, exclusion_zone_mp,
                                                           p=p_mp)
    return _evaluate(motifs_lst, start, end, df_labels)


window_size_mp_e = 10
num_compare_segments_mp_e = 6
max_distance_mp_e = 8.0
exclusion_zone_mp_e = 10
p_mp_e = 1.0


def evaluate_matrix_profile_e_sax(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: window size {window_size_mp_e}")
    print(f"{sax_variant.name}: num compare segments {num_compare_segments_mp_e}")
    print(f"{sax_variant.name}: max distance {max_distance_mp_e}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_mp_e}")
    print(f"{sax_variant.name}: p {p_mp_e}")
    motifs_lst, start, end = do_matrix_profile_discretized(df_motifs, window_size_mp_e, sax_variant,
                                                           num_compare_segments_mp_e, max_distance_mp_e,
                                                           exclusion_zone_mp_e, p=p_mp_e)
    return _evaluate(motifs_lst, start, end, df_labels)


window_size_mp_od = 5
num_compare_segments_mp_od = 24
max_distance_mp_od = 23.0
exclusion_zone_mp_od = 20
p_mp_od = 1.0


def evaluate_matrix_profile_one_d_sax(df_motifs, df_labels, sax_variant):
    print(f"{sax_variant.name}: window size {window_size_mp_od}")
    print(f"{sax_variant.name}: num compare segments {num_compare_segments_mp_od}")
    print(f"{sax_variant.name}: max distance {max_distance_mp_od}")
    print(f"{sax_variant.name}: exclusion zone {exclusion_zone_mp_od}")
    print(f"{sax_variant.name}: p {p_mp_od}")
    motifs_lst, start, end = do_matrix_profile_discretized(df_motifs, window_size_mp_od, sax_variant,
                                                           num_compare_segments_mp_od, max_distance_mp_od,
                                                           exclusion_zone_mp_od, p=p_mp_od)
    return _evaluate(motifs_lst, start, end, df_labels)


len_subsequence_mp_raw = 120
max_distance_mp_raw = 20.0
exclusion_zone_mp_raw = 120
p_mp_raw = 1.0


def evaluate_matrix_profile_raw(df_motifs, df_labels):
    print(f"Raw: len subsequence {len_subsequence_mp_raw}")
    print(f"Raw: max distance {max_distance_mp_raw}")
    print(f"Raw: exclusion zone {exclusion_zone_mp_raw}")
    print(f"Raw: p {p_mp_raw}")
    motifs_lst, start, end = do_matrix_profile_raw(df_motifs, len_subsequence_mp_raw,
                                                   max_distance_mp_raw, exclusion_zone_mp_raw, p=p_mp_raw)
    return _evaluate(motifs_lst, start, end, df_labels)
