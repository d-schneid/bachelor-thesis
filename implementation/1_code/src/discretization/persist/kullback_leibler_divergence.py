import numpy as np


# to avoid relative frequencies of zero in the discrete probability function to
# be able to compute the Kullback-Leibler divergence
SMOOTHING_PARAM = 1e-06


def _smooth(df_prob):
    """
    Add a smoothing parameter to the given probability distribution to avoid
    probabilities of zero when computing the Kullback-Leibler divergence.

    :param df_prob: dataframe of shape (num_bins, num_ts)
        Contains probabilities.
    :return: dataframe of shape (num_bins, num_ts)
    """

    df_prob = df_prob.add(SMOOTHING_PARAM).where(df_prob == 0, df_prob)
    df_prob = df_prob.add(-SMOOTHING_PARAM).where(df_prob == 1, df_prob)

    return df_prob


def _compute_kl_div(df_p, df_q):
    """
    Compute the Kullback-Leibler divergence of the corresponding pairs of the
    given probabilities.

    :param df_p: dataframe of shape (num_bins, num_ts)
        These probabilities are used as the first argument.
    :param df_q: dataframe of shape (num_bins, num_ts)
        These probabilities are used as the second argument.
    :return: dataframe of shape (num_bins, num_ts)
    """

    return df_p * np.log(df_p / df_q)


def _compute_symmetric_kl_div(df_self_transitions, df_self_transitions_complement,
                             df_marginal, df_marginal_complement):
    """
    Compute the symmetric Kullback-Leibler divergence between the transition
    and marginal probability distribution that is build for the respective
    probability and its complement for every state.

    :param df_self_transitions: dataframe of shape (num_bins, num_ts)
        Contains the self transition probability for every state and time
        series.
    :param df_self_transitions_complement: dataframe of shape (num_bins, num_ts)
        Contains the complements of the self transition probabilities for every
        state and time series.
        The complement is computed by: 1 - self transition probability
    :param df_marginal: dataframe of shape (num_bins, num_ts)
        Contains the marginal probability for every state and time series.
    :param df_marginal_complement: dataframe of shape (num_bin, num_ts)
        Contains the complement of the marginal probabilities for every state
        and time series.
        The complement is computed by: 1 - marginal probability.
    :return: dataframe of shape (num_bins, num_ts)
    """

    df_self_transitions = _smooth(df_self_transitions)
    df_self_transitions_complement = _smooth(df_self_transitions_complement)
    df_marginal = _smooth(df_marginal)
    df_marginal_complement = _smooth(df_marginal_complement)

    df_kl_div_fst = (_compute_kl_div(df_self_transitions, df_marginal) +
                     _compute_kl_div(df_self_transitions_complement, df_marginal_complement)) / 2
    df_kl_div_snd = (_compute_kl_div(df_marginal, df_self_transitions) +
                     _compute_kl_div(df_marginal_complement, df_self_transitions_complement)) / 2

    return (df_kl_div_fst + df_kl_div_snd) / 2
