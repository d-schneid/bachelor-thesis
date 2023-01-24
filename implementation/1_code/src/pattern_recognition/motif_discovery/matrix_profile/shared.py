def _find_nearest_neighs(matrix_profile, max_distance):
    """
    Find all subsequences that are nearest neighbors and within 'max_distance'
    of each other.

    :param matrix_profile: np.array of shape (num_subsequences, 2)
        Contains a sub-array for each subsequence where the first entry is the
        distance of the respective subsequence to its nearest neighbor and the
        second entry is the index of its nearest neighbor in this np.array.
    :param max_distance: float
        The upper bound for the distance between two nearest neighbors such
        that they are considered a potential motif.
    :return: dict
        Contains tuples with the indexes of two nearest neighbors as keys and
        the corresponding distance as values. The subsequences corresponding to
        the contained indexes are considered as a potential motif.
    """

    nearest_neighs = {}
    for idx, subsequence in enumerate(matrix_profile):
        nearest_neighbor = matrix_profile[subsequence[1]]
        # 'nearest_neighbor' of 'subsequence' has another neighbor with
        # that it has a lower distance than to 'subsequence'
        # not all subsequences have reciprocal nearest neighbors, because
        # being a nearest neighbor is not symmetric
        # therefore, avoid checking pairs that are not reciprocal nearest
        # neighbors
        # also, avoid checking reciprocal nearest neighbors twice
        if (nearest_neighbor[1], subsequence[1]) in nearest_neighs or\
                (subsequence[1], nearest_neighbor[1]) in nearest_neighs:
            continue

        # reciprocal nearest neighbors
        if nearest_neighbor[1] == idx:
            # reciprocal nearest neighbors are within 'max_distance' of each
            # other
            # distance is symmetric: subsequence[0] == nearest_neighbor[0]
            if subsequence[0] <= max_distance:
                nearest_neighs[(idx, subsequence[1])] = subsequence[0]

    return nearest_neighs
