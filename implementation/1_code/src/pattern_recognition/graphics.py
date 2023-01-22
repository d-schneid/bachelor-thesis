import numpy as np
import matplotlib.pyplot as plt


def plot_eval_metrics_comparison(raw, inverse_transformed, discretized_encoded):
    """
    For each evaluation metric plot two bars. One for the non-discretized
    fitted algorithm and the other for the discretized fitted algorithm.

    :param raw: dict
        Keys are the abbreviations of the used evaluation metrics. Values are
        the corresponding scores of the evaluation metrics computed for the
        algorithm fitted on the raw data.
    :param inverse_transformed: dict
        Keys are the abbreviations of the used evaluation metrics. Values are
        the corresponding scores of the evaluation metrics computed for the
        algorithm fitted on the inverse transformed data.
    :param discretized_encoded: dict
        Keys are the abbreviations of the used evaluation metrics. Values are
        the corresponding scores of the evaluation metrics computed for the
        algorithm fitted on the discretized and encoded data.
    :return:
        None
    """

    # keys are the same for both dicts
    keys = list(raw.keys())
    plt.figure(figsize=(15, 5))
    bar_width = 0.2
    bar_space = 1.3
    num_bars = 3

    plt.bar(keys, raw.values(), width=bar_width, label="raw", edgecolor="black")
    plt.bar(np.arange(len(keys)) + bar_width * bar_space,
            inverse_transformed.values(), width=bar_width,
            label="inverse transformed", edgecolor="black")
    plt.bar(np.arange(len(keys)) + (num_bars - 1) * bar_width * bar_space,
            discretized_encoded.values(), width=bar_width,
            label="discretized encoded", edgecolor="black")

    plt.xticks(np.arange(len(keys)) + num_bars * bar_width * bar_space / num_bars, keys)
    plt.legend()
    plt.show()
