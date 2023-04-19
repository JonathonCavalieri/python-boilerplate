from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier

random_state = 42

# Dataframes with data
X_test = None
X_train = None
y_test = None
y_train = None
EVAL_test = None

class_weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
scores = []
for class_weight in class_weights:
    params = {
        "learning_rate": 0.2,
        "n_estimators": 500,
        "eval_metric": "aucpr",
        "max_depth": 8,
        "objective": "binary:logistic",
        "random_state": random_state,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "scale_pos_weight": class_weight,
    }
    model = XGBClassifier(**params)
    eval_set = [(X_test, y_test)]

    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    preds = model.predict(X_test)
    p = precision_score(y_test, preds)
    r = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    filter_mask = EVAL_test["EVAL_Customer_class_number"] == 3
    class_3_preds = model.predict(X_test[filter_mask])
    class_3_p = precision_score(y_test[filter_mask], class_3_preds)
    class_3_r = recall_score(y_test[filter_mask], class_3_preds)
    class_3_f1 = f1_score(y_test[filter_mask], class_3_preds)

    scores.append((f1, r, p, class_3_f1, class_3_r, class_3_p))
    print(
        f"model with class weight: {class_weight} f1:{f1:.2f} recall: {r:.2f} precision: {p:.2f} | Class Large f1:{class_3_f1:.2f} recall: {class_3_r:.2f} precision: {class_3_p:.2f}"
    )
