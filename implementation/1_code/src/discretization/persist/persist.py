import warnings
import numpy as np
import pandas as pd

from utils import interpolate_segments
from discretization.symbol_mapping import IntervalNormMedian
from discretization.abstract_discretization import AbstractDiscretization, _get_alphabet
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


class Persist(AbstractDiscretization):
    """
    Persist discretization algorithm for time series.

    :param min_alphabet_size: int
        The minimum alphabet size (inclusive) that shall be used to find
        discretization intervals.
    :param max_alphabet_size: int (default = None)
        The maximum alphabet size (inclusive) that shall be used to find
        discretization intervals.
        If None, this will be equal to 'min_alphabet_size'.
    :param skip: int (default = 4)
        The number of breakpoints that shall be ignored for consideration on
        the left and right side of a found breakpoint.
    :raises:
        ValueError: If 'min_alphabet_size' is above 26 or below 1.
        ValueError: If 'min_alphabet_size' > 'max_alphabet_size'.
        ValueError: If 'skip' < 0.

    References
    ----------
    [1] Mörchen, Fabian, and Alfred Ultsch. "Optimizing time series
    discretization for knowledge discovery." Proceedings of the eleventh ACM
    SIGKDD international conference on Knowledge discovery in data mining.
    2005.
    """

    def __init__(self, min_alphabet_size, max_alphabet_size=None, skip=4):
        if max_alphabet_size is not None:
            if min_alphabet_size > max_alphabet_size:
                raise ValueError("The maximum alphabet size needs to be at "
                                 "least as large as the minimum alphabet size.")
        if skip < 0:
            raise ValueError("The 'skip' needs to be at least 0.")

        super().__init__(alphabet_size=min_alphabet_size)
        self.name = "Persist"
        self.min_alphabet_size = min_alphabet_size
        self.max_alphabet_size = min_alphabet_size if max_alphabet_size is None else max_alphabet_size
        self.skip = skip

    def transform(self, df_paa, df_norm, df_breakpoints=None, *args, **kwargs):
        """
        Transform the PAA representation of each time series into its Persist
        representation (i.e. assign each PAA representation its respective
        Persist word).

        :param df_paa: dataframe of shape (num_segments, num_ts)
            The PAA representations of a time series dataset that shall be
            transformed into their Persist representations.
        :param df_norm: dataframe of shape (ts_size, num_ts)
            The original normalized time series dataset corresponding to the
            given PAA representations.
        :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts) (default = None)
            The individual breakpoints for each given PAA representation that
            shall be used to transform the respective PAA representation into
            its Persist representation.
            If None, the respective breakpoints resulting from the Persist
            algorithm on 'df_norm' are used.
            This parameter is intended to allow breakpoints based on the sole
            computation of the Persist algorithm before this transformation.
            For example, to run the Persist algorithm on 'df_paa' instead of
            'df_norm'.
        :return:
            dataframe of shape (num_segments, num_ts):
                The Persist representation for each time series.
            alphabets: list of len = num_ts
                Contains the computed alphabet for each time series as a
                np.array.
            breakpoints: list of len = num_ts
                Contains the respective breakpoints, either computed or given,
                for each time series as a list.
        """

        if df_breakpoints is None:
            breakpoints, best_p_scores = do_persist(df_norm, self.min_alphabet_size,
                                                    self.max_alphabet_size, self.skip)
        else:
            # for further uniform processing
            breakpoints = [list(df_breakpoints.iloc[:, i])
                           for i in range(df_breakpoints.shape[1])]

        persist_ts = []
        alphabets = []
        for idx, curr_breakpoints in enumerate(breakpoints):
            alphabet_idx = np.searchsorted(curr_breakpoints, df_paa.iloc[:, idx], side="right")

            # same number of breakpoints/alphabet for each time series
            if self.min_alphabet_size == self.max_alphabet_size:
                alphabet = self.alphabet
            # time series can have different number of breakpoints/alphabets
            else:
                alphabet_size = len(curr_breakpoints) + 1
                alphabet = _get_alphabet(alphabet_size)

            persist_ts.append(pd.Series(alphabet[alphabet_idx]))
            alphabets.append(alphabet)

        return pd.concat(persist_ts, axis=1), alphabets, breakpoints

    def inv_transform(self, df_persist, ts_size, window_size, symbol_mapping,
                      alphabets, breakpoints, *args, **kwargs):
        """
        Approximate the original time series dataset by transforming its
        Persist representations into a time series dataset with the same size
        by assigning each point the symbol value of its segment based on the
        individual breakpoints of each Persist representation.

        :param df_persist: dataframe of shape (num_segments, num_ts)
            The Persist representation of the time series dataset.
        :param ts_size: int
            The size of the original time series.
        :param window_size: int
            The size of the segments with which the given Persist
            representations were created.
        :param symbol_mapping: SymbolMapping
            The symbol mapping strategy that determines the symbol values for
            the Persist symbols.
        :param alphabets: list of len = num_ts
            Contains the computed alphabet for each time series as a np.array.
        :param breakpoints: list of len = num_ts
            Contains the respective breakpoints, either computed or given, for
            each time series as a list.
        :return: dataframe of shape (ts_size, num_ts)
        """

        if isinstance(symbol_mapping, IntervalNormMedian):
            warnings.warn("Use the chosen 'symbol_mapping' strategy with "
                          "caution, because in the Persist, the breakpoint "
                          "intervals are not determined based on a Gaussian "
                          "distribution.")

        inv_persist_reprs = []
        for idx in range(df_persist.shape[1]):
            curr_persist_ts = df_persist.iloc[:, idx].to_frame()
            curr_breakpoints = np.array(breakpoints[idx])
            curr_alphabet = np.array(alphabets[idx])
            df_mapped = symbol_mapping.get_mapped(curr_persist_ts, curr_alphabet, curr_breakpoints, idx)
            df_inv_persist = interpolate_segments(df_mapped, ts_size, window_size)
            inv_persist_reprs.append(df_inv_persist)

        return pd.concat(inv_persist_reprs, axis=1)

    def transform_inv_transform(self, df_paa, df_norm, window_size, df_breakpoints=None, **symbol_mapping):
        df_persist, alphabets, breakpoints = self.transform(df_paa, df_norm, df_breakpoints)
        return self.inv_transform(df_persist, df_norm.shape[0], window_size,
                                  **symbol_mapping, alphabets=alphabets, breakpoints=breakpoints)

    def transform_to_symbolic_ts(self, df_paa, df_norm, window_size, df_breakpoints=None):
        df_persist, alphabets, breakpoints = self.transform(df_paa, df_norm, df_breakpoints)
        return interpolate_segments(df_persist, df_norm.shape[0], window_size)

    def transform_to_symbolic_repr_only(self, df_paa, df_norm, window_size, df_breakpoints):
        df_persist, alphabets, breakpoints = self.transform(df_paa=df_paa, df_norm=df_norm,
                                                            window_size=window_size,
                                                            df_breakpoints=df_breakpoints)
        return df_persist

    def compute_breakpoints(self, df_norm):
        """
        Compute individual breakpoints for each time series based on the
        Persist algorithm on the normalized time series points.
        This method should only be used if
        'self.min_alphabet_size' == 'self.max_alphabet_size'.

        :param df_norm: dataframe of shape (ts_size, num_ts)
            The time series for that individual breakpoints shall be computed.
        :return: dataframe of shape (num_breakpoints, num_ts)
        """

        breakpoints, best_p_scores = do_persist(df_norm, self.min_alphabet_size,
                                                self.max_alphabet_size, self.skip)
        return pd.DataFrame(zip(*breakpoints))
