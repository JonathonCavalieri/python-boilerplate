from sklearn.metrics import f1_score, recall_score, precision_score
from tabulate import tabulate
import numpy as np

random_state = 42

# Dataframes with data
X_test = None
X_train = None
y_test = None
y_train = None
EVAL_test = None

# XGB Model
model_xgb = None

preds_proba = model_xgb.predict_proba(X_test)


unique_values = np.unique(EVAL_test["EVAL_Customer_class_number"])
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data_f1 = {"class_number": unique_values}
data_recall = {"class_number": unique_values}
data_precision = {"class_number": unique_values}
for threshold in thresholds:
    preds = (preds_proba[:, 1] > threshold).astype(np.int8)
    temp_f1 = []
    temp_recall = []
    temp_precision = []
    for value in unique_values:
        filter_mask = EVAL_test["EVAL_Customer_class_number"] == value

        recall_value = recall_score(y_test[filter_mask], preds[filter_mask])
        temp_recall.append(recall_value)

        f1_value = f1_score(y_test[filter_mask], preds[filter_mask])
        temp_f1.append(f1_value)

        precision_value = precision_score(y_test[filter_mask], preds[filter_mask])
        temp_precision.append(precision_value)

    data_f1[f"threshold_{threshold}"] = temp_f1
    data_recall[f"threshold_{threshold}"] = temp_recall
    data_precision[f"threshold_{threshold}"] = temp_precision

print(f'\n{"*"*55}F1 Scores{"*"*55}\n')
print(tabulate(data_f1, headers="keys", tablefmt="psql"))

print(f'\n{"*"*53}Recall Scores{"*"*53}\n')
print(tabulate(data_recall, headers="keys", tablefmt="psql"))

print(f'\n{"*"*51}Precision Scores{"*"*52}\n')
print(tabulate(data_precision, headers="keys", tablefmt="psql"))
