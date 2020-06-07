#
from sklearn.metrics import r2_score


def r2_adj(y_true, y_pred, dim0, dim1):
    r2 = r2_score(y_true, y_pred)
    result = 1 - (1 - r2) * (dim0 - 1) / (dim0 - dim1 -1)
    return result
