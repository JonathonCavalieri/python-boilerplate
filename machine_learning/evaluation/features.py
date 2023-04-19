from numpy import argsort
import pandas as pd


def sklearn_feature_importance(model, train_data, top=None, print_importance=False):
    importances = model.feature_importances_
    sorted_indices = argsort(importances)[::-1]
    feat_labels = list(train_data.columns)

    if top is None:
        top = len(feat_labels)

    if print_importance:
        for f in range(train_data.shape[1]):
            imp = importances[sorted_indices[f]]
            label = feat_labels[sorted_indices[f]]
            print(
                "%2d) %-*s %f"
                % (
                    f + 1,
                    50,
                    feat_labels[sorted_indices[f]],
                    importances[sorted_indices[f]],
                )
            )

    return {feat_labels[x]: importances[x] for x in sorted_indices[:top]}


def xgb_feature_importance(models):
    frames = []
    for model in models:
        gain = model.get_booster().get_score(importance_type="gain")
        gain = pd.DataFrame.from_dict(gain, orient="index", columns=["gain"])
        frames.append(gain)
    importance = pd.concat(frames)
    importance = importance.groupby(level=0).sum() / len(models)
    importance = importance.sort_values("gain", ascending=False)
    return importance.reset_index(drop=False).rename({"index": "feature_name"}, axis=1)


def lightgbm_feature_importance(model):
    importance_df = (
        pd.DataFrame(
            {
                "feature_name": model.booster_.feature_name(),
                "importance_gain": model.booster_.feature_importance(
                    importance_type="gain"
                ),
                "importance_split": model.booster_.feature_importance(
                    importance_type="split"
                ),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df
