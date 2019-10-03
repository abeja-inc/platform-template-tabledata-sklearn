import numpy as np

from sklearn.metrics import accuracy_score as sk_accuracy_score, roc_auc_score as sk_roc_auc_score, mean_squared_error


def accuracy_score(gold, pred):
    return sk_accuracy_score(gold, pred), 0.0


def roc_auc_score(gold, pred):
    return sk_roc_auc_score(gold, pred), 0.0


def root_mean_squared_error(gold, pred):
    return 0.0, np.sqrt(mean_squared_error(gold, pred))
