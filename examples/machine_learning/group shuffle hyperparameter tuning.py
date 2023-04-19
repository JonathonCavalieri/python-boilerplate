import numpy as np
import pandas as pd
import logging
import sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

input_data = None  # Raw input data
target_name = None  # Target column name
train_test_split_column = None  # Column with train/test split assignmnet
eval_features = None  # Features for evaluation and should be used for training
dimension_features = None  # Dimension features


def transform(data):
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=7)
    split = splitter.split(data, groups=data["Cust. No."])
    train_inds, test_inds = next(split)

    # Split Train data
    X_train = data.iloc[train_inds].copy()
    y_train = X_train[target_name]
    X_train = X_train.drop(columns=[*target_name, train_test_split_column])
    y_train = y_train.values.ravel()

    print(X_train.shape)
    # Sample weights
    weights = {1: 2, 2: 4, 3: 4}
    y_sample_weights = X_train["EVAL_Customer_class_number"].replace(weights)
    y_sample_weights = np.where(y_train == 0, 1, y_sample_weights.values.ravel())

    # Split Test Data
    X_test = data.iloc[test_inds].copy()
    y_test = X_test[target_name]
    X_test = X_test.drop(columns=[*target_name, train_test_split_column])
    y_test = y_test.values.ravel()

    # drop eval features
    X_train.drop(columns=eval_features, inplace=True, errors="ignore")
    X_test.drop(columns=eval_features, inplace=True, errors="ignore")

    # Scale float columns
    float_columns = [
        key
        for key, value in dict(input_data.dtypes).items()
        if pd.api.types.is_float_dtype(value)
    ]
    transformer = RobustScaler()
    X_train[float_columns] = transformer.fit_transform(X_train[float_columns])
    X_test[float_columns] = transformer.transform(X_test[float_columns])

    return X_train, y_train, y_sample_weights, X_test, y_test


def objective_tune(trial: Trial, data) -> float:
    static_param = {
        "n_estimators": 1000,
        "tree_method": "hist",
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
    }

    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 20),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 5, step=0.5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9, step=0.1),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
    }
    # Transform Data
    X_train, y_train, y_sample_weights, X_test, y_test = transform(data)

    eval_set = [(X_test, y_test)]
    # Fit Model
    model = XGBClassifier(**static_param, **param)
    model.fit(X_train, y_train, sample_weight=y_sample_weights, eval_set=eval_set)

    # Predict scores
    preds = model.predict(X_test)
    # Return score for this trial
    return average_precision_score(y_test, preds)


#################################

# Fill the couple missing div
input_data["Div."].fillna(input_data["Div."].mode().values[0], inplace=True)

# One Hot encode dimension colummns
OHE = OneHotEncoder(handle_unknown="ignore")
input_data_OHE = OHE.fit_transform(input_data[dimension_features])
OHE_column_names = OHE.get_feature_names(dimension_features)
input_data[OHE_column_names] = input_data_OHE.todense()
input_data.drop(columns=dimension_features, inplace=True, errors="ignore")

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "wurth_tunning"  # Unique identifier of the study.
log_directory = r"C:\Wurth Churn Project\01 Data\00 Log Files"
storage_name = f"sqlite:///{log_directory}{study_name}.db".format(study_name)
# Create Study
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction="maximize",
    sampler=TPESampler(),
)

print("Parameter Tuning is starting now...")
study.optimize(lambda trial: objective_tune(trial, input_data), n_trials=800)
print(
    "Best trial: score {},\nparams {}".format(
        study.best_trial.value, study.best_trial.params
    )
)
