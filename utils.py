import math

from sklearn.metrics import mean_squared_error


def inverse_root_mean_squared_error(gold, pred):
    return 1.0/(math.sqrt(mean_squared_error(gold, pred)))
