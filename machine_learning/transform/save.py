import pandas as pd
import joblib
import glob


def save_test_to_file(X, y, path, name="test"):
    X.to_pickle(f"{path}/X_{name}.pkl")
    y.to_pickle(f"{path}/y_{name}.pkl")


def load_test_from_file(path, eval_features=None, name="test"):
    X = pd.read_pickle(f"{path}/X_{name}.pkl")
    y = pd.read_pickle(f"{path}/y_{name}.pkl")
    y = y.values.ravel().astype("int")
    eval_test = None

    if eval_features is not None:
        eval_test = X[eval_features]
        X.drop(columns=eval_features, inplace=True, errors="ignore")

    return X, y, eval_test


def print_memory_usage(data, metric="MB", label=""):
    metric_mapping = {"B": 0, "KB": 2, "MB": 2, "GB": 3}
    multiplier = metric_mapping[metric]
    memory = data.memory_usage().sum() / (1024**multiplier)
    print(f"Memory usage {label}: {memory:.2f}{metric}")


def save_models(models, name):
    model_save_path = ""
    for i, model in enumerate(models):
        print(f"saving model_{name}_{i}")
        joblib.dump(model, model_save_path + f"model_{name}_fold_{i}")


def load_models(name, folder):
    models = []
    for model_path in glob.glob(f"{folder}model_{name}_fold_*"):
        model = joblib.load(model_path)
        models.append(model)
    return models
