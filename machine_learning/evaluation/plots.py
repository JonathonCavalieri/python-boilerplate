import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from tabulate import tabulate
from inspect import signature
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

from machine_learning.evaluation.general import (
    cumulative_gain_curve,
    cumulative_lift_gain_df,
)


def plot_lift_curve(
    y_true,
    y_probas,
    title="Lift Curve",
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
):
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate Lift Curve for data with "
            "{} category/ies".format(len(classes))
        )

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0], classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains2 = gains2[1:]

    gains1 = gains1 / percentages
    gains2 = gains2 / percentages

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label="Class {}".format(classes[0]))
    ax.plot(percentages, gains2, lw=3, label="Class {}".format(classes[1]))

    ax.plot([0, 1], [1, 1], "k--", lw=2, label="Baseline")

    ax.set_xlabel("Percentage of sample", fontsize=text_fontsize)
    ax.set_ylabel("Lift", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid(True)  # ax.grid("on")
    ax.legend(loc="lower right", fontsize=text_fontsize)

    return ax


def plot_cumulative_gain(
    y_true,
    y_probas,
    title="Cumulative Gains Curve",
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    plot_zero=True,
):
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate Cumulative Gains for data with "
            "{} category/ies".format(len(classes))
        )

    # Compute Cumulative Gain Curves

    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if plot_zero:
        percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0], classes[0])
        ax.plot(percentages, gains1, lw=3, label="Class {}".format(classes[0]))

    ax.plot(percentages, gains2, lw=3, label="Class {}".format(classes[1]))

    ax.set_xlim((0.0, 1.0))  # ax.set_xlim([0.0, 1.0])
    ax.set_ylim(0.0, 1.0)  # ax.set_ylim([0.0, 1.0])

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Baseline")

    ax.set_xlabel("Percentage of sample", fontsize=text_fontsize)
    ax.set_ylabel("Gain", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid(True)  # ax.grid("on")
    ax.legend(loc="lower right", fontsize=text_fontsize)

    return ax


def cumulative_gain_by_dimension(
    dataframe, column, y_test, y_proba, width=2, ax_size=(4, 6), plot_zero=True
):
    unique_values = np.unique(dataframe[column])
    height = ceil(len(unique_values) / width)
    unique_values.resize(width * height)
    unique_values = unique_values.reshape((-1, width))

    # print(unique_values)
    ax_height, ax_width = ax_size

    fig = plt.figure(
        constrained_layout=True, figsize=(ax_width * width, ax_height * height)
    )
    ax = fig.subplots(nrows=height, ncols=width)
    fig.suptitle("Cumulative Gains Curve", fontsize=14)
    for y, row in enumerate(unique_values):
        for x, value in enumerate(row):
            if value == "" or value is None or (value == 0 and y == 1 and x == 1):
                continue

            if height == 1:
                coord = x
            else:
                coord = (y, x)
            filter_mask = dataframe[column] == value
            title = f"{column} : {value}"
            plot_cumulative_gain(
                y_test[filter_mask],
                y_proba[filter_mask],
                ax=ax[coord],
                title=title,
                plot_zero=plot_zero,
            )

    return ax


def eval_model_classification_report(
    x_test,
    y_test,
    y_proba=None,
    models=None,
    custom_scorer_func=None,
    custom_scorer_name="Custom Scorer",
    threshold=0.5,
    title_fontsize="large",
    text_fontsize="medium",
    plot_zero=True,
):
    actual = np.array(y_test).squeeze()

    if models:
        if isinstance(models, list):
            y_score = np.zeros(len(x_test))
            for model in models:
                y_score += model.predict_proba(x_test)[
                    :, 1
                ]  # needs to be updated to proba for multiple models
            # weight each one equally
            y_score /= len(models)
        else:
            y_proba = models.predict_proba(x_test)
            y_score = y_proba[:, 1]

    if y_proba is None:
        raise ValueError("models or y_proba parameter should be filled in")
    y_score = y_proba[:, 1]
    y_preds = (y_score > threshold).astype(np.int8)
    y_score_rounded = np.floor(y_score * 100) / 100

    precision, recall, _ = precision_recall_curve(y_test, y_score_rounded)
    average_precision = average_precision_score(y_test, y_score)
    f1_score_value = f1_score(y_test, y_preds)
    log_loss_value = log_loss(y_test, y_score)
    accuracy_score_value = accuracy_score(y_test, y_preds)
    AUC_ROC = roc_auc_score(y_test, y_score)

    print(f'\n{"*"*26}Summary Scores{"*"*25}\n')
    print(f"{'F1 Score' : <30}{f1_score_value :>5.2f}")
    print(f"{'Average Precision/AUPR Score' : <30}{average_precision :>5.2f}")
    print(f"{'Log Loss Score' : <30}{log_loss_value :>5.2f}")
    print(f"{'Accuracy Score' : <30}{accuracy_score_value :>5.2f}")
    print(f"{'AUROC Score' : <30}{AUC_ROC :>5.2f}")

    if custom_scorer_func:
        custom_score = custom_scorer_func(actual, y_preds)
        print(f"{custom_scorer_name : <30}{custom_score :>5.2f}")

    print(f'\n{"*"*22}Classification Report{"*"*22}\n')
    print(classification_report(y_test, y_preds, target_names=["Negative", "Positive"]))
    print(f'\n{"*"*22}Confusion Matrix{"*"*22}\n')
    # print(confusion_matrix(y_test, y_preds))
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_preds),
        index=["true:no", "true:yes"],
        columns=["pred:no", "pred:yes"],
    )
    print(tabulate(cmtx, headers="keys", tablefmt="psql"))
    print(f'\n{"*"*14}Precision Recall Curve & Cummulative Gains Chart{"*"*14}\n')
    fig = plt.figure(constrained_layout=True, figsize=(12, 4))
    ax = fig.subplots(nrows=1, ncols=2)

    step_kwargs = (
        {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    )
    ax[0].step(recall, precision, color="b", alpha=0.2, where="post")
    ax[0].fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel("Recall", fontsize=text_fontsize)
    ax[0].set_ylabel("Precision", fontsize=text_fontsize)
    ax[0].set_title(f"Precision-Recall curve: AP={average_precision:0.2f}")
    plot_cumulative_gain(y_test, y_proba, ax=ax[1], plot_zero=plot_zero)

    plt.show()
    print(f'\n{"*"*22}Cumulative Gains and Lift Table{"*"*22}\n')
    max_width = dict(
        selector="th", props=[("max-width", "130px"), ("text-align", "center")]
    )
    formating = {
        "Cumulitive % of Customers": "{:.0f}%",
        "Cumulitive % of Dormant Customers": "{:.1f}%",
        "Lift": "{:.1f}",
    }
    df = cumulative_lift_gain_df(y_test, y_proba)
    df = (
        df.style.set_properties(**{"width": "130px", "text-align": "center"})
        .set_table_styles([max_width])
        .hide_index()
        .format(formating)
    )
    print(tabulate(df, headers="keys", tablefmt="psql"))


def eval_model_classification_report_basic(x_test, y_test, models, scorer_func):
    actual = np.array(y_test).squeeze()

    if isinstance(models, list):
        preds = np.zeros(len(x_test))
        for model in models:
            preds += model.predict_proba(x_test)[:, 1]
        # weight each one equally
        preds /= len(models)
        y_preds = (preds > 0.5).astype(np.int8)
    else:
        preds = models.predict_proba(x_test)[:, 1]
        y_preds = models.predict(x_test)

    score = scorer_func(actual, preds)

    print(f"Validation Results - Metric: {score:.3f} \n\nClassification Report\n")
    print(classification_report(y_test, y_preds, target_names=["Negative", "Positive"]))
    print("\n\nConfusion Matrix\n")
    #     print(confusion_matrix(y_test, y_pred))
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_preds),
        index=["true:yes", "true:no"],
        columns=["pred:yes", "pred:no"],
    )
    print(tabulate(cmtx, headers="keys", tablefmt="psql"))


def classification_evaluation_by_dimensions(dataframe, y_test, y_preds, columns):
    dataframe = dataframe.copy()
    dataframe["prediction"] = y_preds
    dataframe["actual"] = y_test

    dataframe["TP"] = (
        dataframe["actual"] * dataframe["prediction"]
    )  # if P & P = 1, P & N = 0, N & P = 0, N & N = 0
    dataframe["TN"] = (1 - dataframe["actual"]) * (
        1 - dataframe["prediction"]
    )  # if P & P = 0, P & N = 0, N & P = 0, N & N = 1
    dataframe["FN"] = dataframe["actual"] * (
        1 - dataframe["prediction"]
    )  # if P & P = 0, P & N = 1, N & P = 0, N & N = 0
    dataframe["FP"] = (1 - dataframe["actual"]) * dataframe[
        "prediction"
    ]  # if P & P = 0, P & N = 0, N & P = 1, N & N = 0

    for column in columns:
        print(f'\n{"*"*22}{column} report{"*"*22}')
        if column not in dataframe.columns:
            print(f"{column} not in evaluation dataset")
            continue
        agg_columns = [column, "prediction", "actual", "TP", "FP", "TN", "FN"]
        summary = dataframe[agg_columns].groupby(column).agg("sum")
        summary["Recall"] = summary["TP"] / summary["actual"]
        summary["Precision"] = summary["TP"] / summary["prediction"]
        summary["FOR"] = summary["TN"] / (summary["TN"] + summary["FN"])
        summary["fall out"] = summary["FP"] / (summary["TN"] + summary["FP"])

        print(tabulate(summary, headers="keys", tablefmt="psql"))
