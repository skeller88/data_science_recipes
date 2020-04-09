import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer


def log_rmse(ytrue, ypred):
    return np.sqrt(mean_squared_error(np.log(ytrue), np.log(ypred)))


log_rmse_scorer = make_scorer(log_rmse, greater_is_better=False)