import numpy as np
from sklearn.metrics import fbeta_score, f1_score


def f2_score_xgb_inverse(actual, preds_proba):
    preds = (preds_proba > 0.5).astype(np.int8)
    f2 = fbeta_score(actual, preds, beta=2)
    return 1 - f2


def f1_score_xgb_inverse(actual, preds_proba):
    preds = (preds_proba > 0.5).astype(np.int8)
    f1 = f1_score(actual, preds)
    return 1 - f1
