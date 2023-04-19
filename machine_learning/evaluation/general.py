import pandas as pd
import numpy as np


def cumulative_lift_gain_df(y_test, y_proba, percentiles=None):
    percentages, gains = cumulative_gain_curve(y_test, y_proba[:, 1], 1)
    percentages_lift = percentages[1:]
    lift = gains[1:]
    lift = lift / percentages_lift

    if percentiles is None:
        percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    lift_list = []
    gains_list = []

    for p in percentiles:
        lift_list.append(lift[percentages_lift > p][0])
        gains_list.append(gains[percentages > p][0] * 100)

    percentiles = np.multiply(percentiles, 100)

    return_data = pd.DataFrame([percentiles, gains_list, lift_list])
    return_data = return_data.transpose()
    return_data.columns = [
        "Cumulitive % of Customers",
        "Cumulitive % of Dormant Customers",
        "Lift",
    ]

    return return_data


def cumulative_gain_curve(y_true, y_score, pos_label=None):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains
