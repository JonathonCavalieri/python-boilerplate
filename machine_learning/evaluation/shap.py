import numpy as np
import pandas as pd
import math


def top_shap_values(shap_values, input_data, n_top=None):
    feature_names = input_data.columns
    feature_names2 = np.array(feature_names)
    records = len(input_data)
    if n_top is None:
        n_top = len(feature_names)

    # get the index of the top features
    top_index = np.flip(np.argsort(np.abs(shap_values))[:, -n_top:], 1)
    # top feature names
    top_features = [feature_names2[x] for x in top_index]
    # top actual values
    top_actuals = input_data[feature_names].to_numpy()
    top_actuals = np.take_along_axis(top_actuals, top_index, 1)
    # top shap strength values
    top_values = np.take_along_axis(shap_values, top_index, 1)

    remaining_value = shap_values.sum(1) - top_values.sum(1)

    # Combine shap top values into one array
    shap_results = np.dstack((top_features, top_values, top_actuals)).reshape(
        records, -1
    )

    del top_actuals, top_index, top_features

    # Dynamic column name for shap top
    col_str_name = []
    for i in range(n_top):
        i = i + 1
        column_name = f"explaination_{i}_feature_name"
        strength = f"explaination_{i}_shap"
        actuals = f"explaination_{i}_actual_value"
        col_str_name.extend([column_name, strength, actuals])

    # Convert to dataframe
    shap_results = pd.DataFrame(shap_results, columns=col_str_name)
    convert_numeric = {
        x: "float32" for x in shap_results.columns if "feature_name" not in x
    }
    shap_results = shap_results.astype(convert_numeric, copy=False)

    shap_results["remaining_shap_value"] = remaining_value

    return shap_results


def get_shap_values_additive(explainer, input_data, prediction, feature_names):
    shap_values = explainer.shap_values(input_data, approximate=True)

    feature_names2 = np.array(feature_names)
    n_top = len(feature_names)
    records = len(input_data)

    # Get Additive Values
    sum_strength = np.sum(shap_values, 1)
    shap_base_value = explainer.expected_value
    base_value = math.exp(explainer.expected_value) / (
        1 + math.exp(explainer.expected_value)
    )
    shap_diff = prediction[:, 1] - base_value
    shap_add_value = (shap_values / sum_strength.reshape(-1, 1)) * shap_diff.reshape(
        -1, 1
    )

    # get the index of the top features
    top_index = np.flip(np.argsort(np.abs(shap_values))[:, -n_top:], 1)
    # top feature names
    top_features = [feature_names2[x] for x in top_index]
    # top actual values
    top_actuals = input_data[feature_names].to_numpy()
    top_actuals = np.take_along_axis(top_actuals, top_index, 1)
    # top shap strength values
    top_values = np.take_along_axis(shap_values, top_index, 1)
    # top shap additive values
    top_shap_add_value = np.take_along_axis(shap_add_value, top_index, 1)

    # Combine shap top values into one array
    shap_results = np.dstack(
        (top_features, top_values, top_actuals, top_shap_add_value)
    ).reshape(records, -1)

    del top_actuals, top_index, top_features, top_shap_add_value

    # Dynamic column name for shap top
    col_str_name = []
    for i in range(n_top):
        i = i + 1
        column_name = f"EXPLANATION_{i}_FEATURE_NAME"
        strength = f"EXPLANATION_{i}_STRENGTH"
        actuals = f"EXPLANATION_{i}_ACTUAL_VALUE"
        adds = f"EXPLANATION_{i}_ADDITIVE"
        col_str_name.extend([column_name, strength, actuals, adds])

    # Convert to dataframe
    shap_results = pd.DataFrame(shap_results, columns=col_str_name)
    convert_numeric = {
        x: "float32" for x in shap_results.columns if "FEATURE_NAME" not in x
    }
    shap_results = shap_results.astype(convert_numeric, copy=False)
    shap_results["shap_base_value"] = shap_base_value
    shap_results["base_value"] = base_value
    shap_results["sum_strength"] = sum_strength
    shap_results["diff"] = shap_diff

    return shap_results
