from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import numpy as np

model_save_path = ""
random_state = 42


def kfold_training(
    model,
    x_input,
    y_input,
    print_messages=True,
    fit_params={},
    splits=5,
    first_fold_only=False,
):
    skf = StratifiedKFold(n_splits=splits, random_state=random_state, shuffle=True)
    models = []
    scores = []
    if print_messages:
        print("starting cross validation training")
    for fold, (train_index, test_index) in enumerate(skf.split(x_input, y_input)):
        if print_messages:
            print("*" * 100)
            print(" " * 20 + f"Fold: {fold}")
            print("*" * 100)
        x_train, y_train = x_input.iloc[train_index], y_input.iloc[train_index]
        x_test, y_test = x_input.iloc[test_index], y_input.iloc[test_index]

        model_iter = clone(model)
        model_iter.fit(x_train, y_train, eval_set=[(x_test, y_test)], **fit_params)
        preds = model_iter.predict_proba(x_test)[:, 1]

        score = amex_metric_numpy(y_test.to_numpy(), preds)
        scores.append(score)
        models.append(model_iter)
        joblib.dump(model_iter, model_save_path + f"model_temp_training_fold_{fold}")

        if print_messages:
            eval_model_classification_report(x_test, y_test, model_iter)
            print("\n" * 3)

        if first_fold_only:
            break

    overall_score = np.array(scores).mean()

    if print_messages:
        print(f"Cross Validation score: {overall_score}")

    return overall_score, models


def lgbm_modelling(
    x_train,
    y_train,
):
    def custom_metric_lgbm(actual, preds):
        return "amex_metric", amex_metric_numpy(actual, preds), True

    fixed_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "dart",
        #'device': 'gpu',
        "histogram_pool_size": 2000,
    }

    search_params = {
        "num_leaves": 100,
        "learning_rate": 0.01,
        "colsample_bytree": 0.20,
        "subsample_freq": 10,
        "subsample": 0.50,
        "reg_lambda": 2,
        "min_child_samples": 40,
        "n_estimators": 10500,
    }

    model = LGBMClassifier(**fixed_params, **search_params)
    fit_params = {
        "callbacks": [log_evaluation(period=1000)],
        "eval_metric": custom_metric_lgbm,
    }

    score, models = kfold_training(model, x_train, y_train, fit_params=fit_params)


def objective_tune(trial: Trial, x_train, y_train, x_test, y_test) -> float:

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2600, log=True),
        "tree_method": "gpu_hist",
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_discrete_uniform("gamma", 0, 5, 0.5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.05, 0.3),
        "colsample_bytree": trial.suggest_discrete_uniform(
            "colsample_bytree", 0.5, 0.9, 0.1
        ),
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "eta": trial.suggest_loguniform("eta", 1e-8, 1.0),
        "n_jobs": -1,
        "eval_metric": custom_metric,
        "objective": "binary:logistic",
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 5),
        "random_state": 2022,
    }
    model = XGBClassifier(**param)

    model.fit(x_train, y_train)

    preds = model.predict_proba(x_test)[:, 1]
    actual = np.array(y_test).squeeze()

    return a_amex_helper.amex_metric_numpy(actual, preds)


# Example usage
print("Parameter Tuning is starting now...")
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(),
)
study.optimize(
    lambda trial: objective_tune(trial, x_train, y_train, x_test, y_test), n_trials=50
)
print(
    "Best trial: score {},\nparams {}".format(
        study.best_trial.value, study.best_trial.params
    )
)
