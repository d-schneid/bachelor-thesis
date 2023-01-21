import numpy as np
import matplotlib.pyplot as plt


def plot_eval_metrics_comparison(non_discretized, discretized):
    """
    For each evaluation metric plot two bars. One for the non-discretized
    fitted algorithm and the other for the discretized fitted algorithm.

    :param non_discretized: dict
        Keys are the abbreviations of the used evaluation metrics. Values are
        the corresponding scores of the evaluation metrics computed for the
        non-discretized fitted algorithm.
    :param discretized: dict
        Keys are the abbreviations of the used evaluation metrics. Values are
        the corresponding scores of the evaluation metrics computed for the
        discretized fitted algorithm.
    :return:
        None
    """

    # keys are the same for both dicts
    keys = list(non_discretized.keys())
    plt.figure(figsize=(15, 5))
    bar_width = 0.3
    bar_space = 1.2
    num_bars = 2
    plt.bar(keys, non_discretized.values(), width=bar_width,
            label="non_discretized", edgecolor="black")
    plt.bar(np.arange(len(keys)) + bar_width * bar_space, discretized.values(),
            width=bar_width, label='discretized', edgecolor="black")
    plt.xticks(np.arange(len(keys)) + bar_width * bar_space / num_bars, keys)
    plt.legend()
    plt.show()
