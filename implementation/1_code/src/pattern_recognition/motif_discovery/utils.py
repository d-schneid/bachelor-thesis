from discretization.sax.one_d_sax import OneDSAX


def _encode_symbols(symbol_split, sax_variant):
    """
    Encode SAX symbols to numbers. Symbol 'a' is assigned 0, symbol 'b' is
    assigned 1 and so on.

    :param symbol_split: list of len(symbol_split) = num_ts
        Contains lists of the SAX representations for each time series,
        but split according to the number of symbols per segment for the given
        'sax_variant'. Hence, each sub-list contains one dataframe for each
        number of symbols per segment.
    :param sax_variant: AbstractSAX
        The SAX variant that was used to create the given SAX representations.
    :return:
        list of len(symbol_split_encoded) = num_ts
            Has the same structure as the given list 'symbol_split', but with
            encoded symbols.
    """

    mapping = dict(zip(sax_variant.alphabet,
                       [num for num in range(sax_variant.alphabet_size)]))
    if isinstance(sax_variant, OneDSAX):
        if sax_variant.alphabet_size_slope > sax_variant.alphabet_size:
            mapping = dict(zip(sax_variant.alphabet_slope,
                               [num for num in range(sax_variant.alphabet_size_slope)]))

    symbol_split_encoded = []
    for sax_word_split in symbol_split:
        symbol_split_encoded.append([sax_symbol_split.replace(to_replace=mapping)
                                     for sax_symbol_split in sax_word_split])

    return symbol_split_encoded


def _remove_trivial(motif, exclusion_zone):
    """
    Remove trivial matches in given motif.

    :param motif: list
        Contains indexes corresponding to subsequences that form a motif.
    :param exclusion_zone: int
        The number of indexes on the left and right side of an index in the
        given motif that shall be excluded from motif discovery.
    :return:
        cleaned_motif: list
            The given motif with trivial mathces removed.
        trivial:
            The indexes that are in the exclusion zones of the indexes in
            'cleaned_motif'.
    """

    cleaned_motif = []
    trivial = []
    current_trivial = []
    for idx in motif:
        if idx not in current_trivial:
            cleaned_motif.append(idx)
            # exclusion zone around subsequence in motif
            current_trivial = [idx + i for i in range(1, exclusion_zone + 1)] +\
                              [idx - i for i in range(1, exclusion_zone + 1)]
            trivial.extend(current_trivial)

    # given motif contains only trivials that were removed
    if len(cleaned_motif) == 1:
        cleaned_motif = []
        # if there is no motif, there are no trivials
        trivial = []

    return cleaned_motif, trivial


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
        # sort in ascending order by the indexes of the subsequences
        motifs_per_subsequence_lst.append(dict(sorted(motifs_per_subsequence.items(),
                                                      key=lambda tup: tup[0])))

    return motifs_per_subsequence_lst


def remove_trivial_motifs_per_subsequence(motifs_per_subsequence_lst):
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
