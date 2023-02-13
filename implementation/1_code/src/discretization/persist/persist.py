import numpy as np
import pandas as pd

from discretization.persist.persistence_score import _compute_persistence_score


def _select_best_breakpoints(min_alphabet_size, best_p_scores, best_breakpts_ts):
    """
    Select the breakpoints with the highest corresponding persistence score
    for each time series.

    :param min_alphabet_size: int
        The minimum alphabet size (inclusive) that shall be considered for
        finding the best discretization intervals.
    :param best_p_scores: dataframe of shape ('max_alphabet_size - 'min_alphabet_size' + 1, num_ts)
        The maximum persistence score for each considered alphabet size for
        each time series.
    :param best_breakpts_ts: list of len = num_ts
        Contains a 2d-list for each time series. Such a 2d-list contains the
        best breakpoints for each considered alphabet size and is sorted by
        ascending alphabet sizes and ascending breakpoints within the
        corresponding 1d-lists.
    :return: list of len = num_
        Contains a list of the breakpoints that correspond to the maximum
        persistence score for each time series across the considered alphabet
        sizes. These breakpoints shall be used for discretization according to
        the Persist algorithm.
    """

    discretization_breakpts = []
    # dataframe index starts with 'min_alphabet_size'
    best_breakpt_idxs = best_p_scores.idxmax(axis=0) - min_alphabet_size
    discretization_breakpts += [best_breakpts_ts[idx_ts][idx_best_breakpt]
                                for idx_ts, idx_best_breakpt in enumerate(best_breakpt_idxs)]

    return discretization_breakpts


def do_persist(df_norm, min_alphabet_size, max_alphabet_size, skip):
    """
    Compute the best discretization intervals for alphabet sizes in the range
    ['min_alphabet_size', 'max_alphabet_size'] for every time series based on
    the Persist algorithm.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series that shall be discretized.
    :param min_alphabet_size: int
        The minimum alphabet size (inclusive) that shall be considered for
        finding the best discretization intervals.
    :param max_alphabet_size: int
        The maximum alphabet size (inclusive) that shall be considered for
        finding the best discretization intervals.
    :param skip: int
        The number of breakpoints that shall be ignored for consideration on
        the left and right side of a found breakpoint.
    :return:
        discretization_breakpts: list of len = num_ts
            Contains a list of the breakpoints that correspond to the maximum
            persistence score for each time series across the considered
            alphabet sizes. These breakpoints shall be used for discretization
            according to the Persist algorithm.
        best_p_scores: dataframe of shape ('max_alphabet_size' - 'min_alphabet_size' + 1, num_ts)
            Contains the persistence score for each considered alphabet size
            for each time series.
    """

    # percentiles of time series data
    candidate_breakpts = pd.DataFrame(np.percentile(df_norm, [i for i in range(1, 100)], axis=0))
    free_breakpts = pd.DataFrame(np.full((candidate_breakpts.shape[0], candidate_breakpts.shape[1]), True))
    free_breakpts.iloc[:skip, :] = False
    free_breakpts.iloc[-skip:, :] = False

    best_breakpts_ts = []
    best_p_scores = pd.DataFrame(np.zeros((max_alphabet_size - min_alphabet_size + 1,
                                          df_norm.shape[1])), index=range(min_alphabet_size, max_alphabet_size + 1))
    for i in range(df_norm.shape[1]):
        best_breakpts = []
        current_ts = df_norm.iloc[:, i].to_frame()
        current_candidate_breakpts = candidate_breakpts.iloc[:, i]
        current_free_breakpts = free_breakpts.iloc[:, i]
        row = 0

        # in every iteration, one breakpoint is added
        for num_breakpoints in range(1, max_alphabet_size):
            current_breakpts = current_free_breakpts[current_free_breakpts == True].index
            if current_breakpts.size <= 0:
                continue

            # persistence score when adding every free candidate breakpoint
            p_scores_breakpts = []
            for idx in current_breakpts:
                breakpoints = sorted(best_breakpts + [current_candidate_breakpts[idx]])
                persistence_score = _compute_persistence_score(current_ts, pd.DataFrame(breakpoints))
                p_scores_breakpts.append(persistence_score)

            best_p_score = np.amax(p_scores_breakpts)
            best_idx = np.argmax(p_scores_breakpts)
            best_breakpt = current_candidate_breakpts[current_breakpts[best_idx]]
            best_breakpts.append(best_breakpt)
            # do not consider breakpoints near selected best breakpoints
            current_free_breakpts[current_breakpts[best_idx] - skip:current_breakpts[best_idx] + skip + 1] = False

            # num_breakpoints corresponds to num_breakpoints + 1 bins
            if num_breakpoints + 1 >= min_alphabet_size:
                best_p_scores.iloc[row, i] = best_p_score
                row += 1

        # only get breakpoints that correspond to considered alphabet sizes
        best_breakpts_per_alph = [best_breakpts[:i] for i in range(min_alphabet_size - 1, max_alphabet_size)]
        best_breakpts_per_alph = [sorted(sublist) for sublist in best_breakpts_per_alph]
        best_breakpts_ts.append(best_breakpts_per_alph)

    discretization_breakpoints = _select_best_breakpoints(min_alphabet_size, best_p_scores, best_breakpts_ts)
    return discretization_breakpoints, best_p_scores
