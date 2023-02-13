import pandas as pd


def _compute_markov_chain_max_likelihood(df_discretized, num_bins):
    """
    Compute the marginal and transition probabilities of the states (i.e. bins)
    based on the maximum likelihood estimation for Markov chains. For the
    transition probabilities, a first-order Markov model is used. Hence, the
    probability to transition into a state only depends on the previous state.

    :param df_discretized: dataframe of shape (ts_size, num_ts)
        The time series points with their corresponding bin ids.
    :param num_bins: int
        The number of bins that are used for discretization.
    :return:
        df_marginal: dataframe of shape (num_bins, num_ts)
            Contains the marginal probability for each state in each time
            series.
        transitions: list of len(transitions) = num_ts
            Contains a dataframe of shape (num_bins, num_bins) for each time
            series. Such a dataframe contains the transition probabilities from
            one state to another. The 'from' states are represented by the rows
            and the 'to' states are represented by the columns.
    """

    df_absolute = df_discretized.apply(lambda ts: ts.value_counts().sort_index())
    df_marginal = df_absolute / df_discretized.shape[0]
    # if a time series does not contain every state; but every given state
    # needs to be represented
    df_marginal = df_marginal.reindex(index=range(num_bins)).fillna(0)

    transitions = []
    for i in range(df_discretized.shape[1]):
        current_ts = list(df_discretized.iloc[:, i])
        df_transitions = pd.crosstab(index=current_ts[:-1], columns=current_ts[1:],
                                     values=1, rownames=["from"], colnames=["to"],
                                     aggfunc="sum", dropna=False, normalize="index")
        # make sure every state is contained
        df_transitions = df_transitions.reindex(index=range(num_bins)).fillna(0)
        df_transitions = df_transitions.T.reindex(index=range(num_bins)).fillna(0).T
        transitions.append(df_transitions)

    return df_marginal, transitions
