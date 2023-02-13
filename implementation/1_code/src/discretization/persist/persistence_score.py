import numpy as np
import pandas as pd

from discretization.persist.discretize import _discretize
from discretization.persist.markov_chain_max_likelihood import _compute_markov_chain_max_likelihood
from discretization.persist.kullback_leibler_divergence import _compute_symmetric_kl_div


def _compute_persistence_score(df_norm, df_breakpoints):
    """
    Compute the mean persistence score across all states for every time series.

    :param df_norm: dataframe of shape (ts_size, num_ts)
        The time series that shall be discretized.
    :param df_breakpoints: dataframe of shape (num_breakpoints, num_ts)
        The breakpoints from which the discretization intervals shall be built
        for every time series.
    :return: pd.Series of shape (num_ts,)
    """

    df_discretized = _discretize(df_norm, df_breakpoints)

    df_marginal, transitions = _compute_markov_chain_max_likelihood(df_discretized,
                                                                    df_breakpoints.shape[0] + 1)

    self_transitions = [pd.Series(np.diag(transition)) for transition in transitions]
    df_self_transitions = pd.concat(self_transitions, axis=1)
    df_marginal_complement = 1 - df_marginal
    df_self_transitions_complement = 1 - df_self_transitions
    df_kldiv = _compute_symmetric_kl_div(df_self_transitions, df_self_transitions_complement,
                                         df_marginal, df_marginal_complement)

    persistence_score = ((-1)**(df_self_transitions < df_marginal) * df_kldiv).mean()

    return persistence_score
