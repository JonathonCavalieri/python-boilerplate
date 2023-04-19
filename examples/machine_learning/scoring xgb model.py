#################################
import pandas as pd
import numpy as np
import re
import gc

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBClassifier
import shap
from joblib import load


#################################
save_directory = ""


#################################
def f2_score_xgb_inverse(actual, preds_proba):
    preds = (preds_proba > 0.5).astype(np.int8)
    f2 = fbeta_score(actual, preds, beta=2)
    return 1 - f2


def get_top_shap_values(shap_values, input_data, n_top=None):
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
    gc.collect()

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


#################################
print("Scoring Pipeline - Reading Data")
input_data = Alteryx.read("#1")
# Create a copy of row id for output
output_dataframe = input_data[["Cust. No.", "date_month"]].copy()


#################################
print("Scoring Pipeline - Loading Artifacts")
column_metadata = load(f"{save_directory}\column_metadata.joblib")

OHE = load(f"{save_directory}\one_hot_encoder.joblib")
transformer = load(f"{save_directory}\scaler.joblib")
model_xgb = load(f"{save_directory}\model.joblib")
explainer = load(f"{save_directory}\shap_explainer.joblib")


#################################
# Read metadata from traing workflow output dictionary
training_columns = column_metadata["training_columns"]
training_columns_dtypes = column_metadata["training_columns_dtypes"]
float_columns = column_metadata["float_columns"]
dimension_columns = column_metadata["dimension_columns"]
div_impute_value = column_metadata["div_impute_value"]


#################################
print("Scoring Pipeline - Preping data")
# fix any column names that have bad chars
input_data.columns = [re.sub(',|[|]|{|}|:|"', "", x) for x in input_data.columns]

drop_columns = [x for x in input_data.columns if x not in training_columns]
input_data.drop(columns=drop_columns, inplace=True, errors="ignore")

# impute missing values
input_data["Div."].fillna(div_impute_value, inplace=True)

# encode string fields
OHE_column_names = OHE.get_feature_names(dimension_columns)
input_data[OHE_column_names] = OHE.transform(input_data[dimension_columns]).todense()
input_data.drop(columns=dimension_columns, inplace=True, errors="ignore")

# keep a copy of the unscaled data for output
unscaled_input_data = input_data.copy()

# scale float fields
input_data[float_columns] = transformer.transform(input_data[float_columns])

# make sure correct dtype
input_data = input_data.astype(training_columns_dtypes, copy=False)


#################################
print("Scoring Pipeline - Scoring data")
output_dataframe[
    ["nondormant_probability", "dormant_probability"]
] = model_xgb.predict_proba(input_data)


#################################
print("Scoring Pipeline - Calculating shap values")
shap_values = explainer(input_data)


#################################
shap_values_class_1 = shap_values.values[
    :, :, 1
]  # just get the shap values for class 1
output_dataframe["base_value"] = shap_values.base_values[:, 1]
top_features = get_top_shap_values(shap_values_class_1, unscaled_input_data, n_top=10)
output_dataframe = pd.concat([output_dataframe, top_features], axis=1)


#################################
output_dataframe_shap = output_dataframe[["Cust. No.", "date_month"]].copy()
output_dataframe_shap[input_data.columns] = shap_values_class_1
output_dataframe_shap = output_dataframe_shap.melt(
    id_vars=["Cust. No.", "date_month"], var_name="column_name", value_name="SHAP_value"
)
output_dataframe_shap["rank"] = (
    output_dataframe_shap["SHAP_value"]
    .abs()
    .groupby(output_dataframe_shap["Cust. No."])
    .rank(ascending=False, method="min")
)


output_dataframe_shap["original_value"] = unscaled_input_data.melt()["value"]
output_dataframe_shap["scaled_value"] = input_data.melt()["value"]
