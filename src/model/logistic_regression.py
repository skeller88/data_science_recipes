import numpy as np
import pandas as pd
from scipy import stats


class LogisticRegression(linear_model.LogisticRegression):
    def fit(self):
        sse = sum((y - predictions) ** 2)
        MSE = sse / (len(newX) - len(newX.columns))

        # Note if you don't want to use a DataFrame replace the two lines above with
        # newX = np.append(np.ones((len(X),1)), X, axis=1)
        # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)

        myDF3 = pd.DataFrame()
        myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b, p_values]
        print(myDF3)

#
# class LogisticRegression(linear_model.LogisticRegression):
#     """
#     From https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
#
#     TODO - sanity check against:
#     https://www.statsmodels.org/stable/_modules/statsmodels/regression/linear_model.html
#
#     LinearRegression class after sklearn's, but calculate t-statistics and p-values for model coefficients (betas).
#     Additional attributes available after .fit() are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
#     which is (n_features, n_coefs)/
#
#     This class sets the intercept to 0 by default, since usually we include it in X.
#     """
#
#     def __init__(self, *args, **kwargs):
#         if not "fit_intercept" in kwargs:
#             kwargs['fit_intercept'] = False
#         super(LogisticRegression, self).__init__(*args, **kwargs)
#
#     def fit(self, X, y, n_jobs=1):
#         self = super(LogisticRegression, self).fit(X, y, n_jobs)
#
#         sse = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
#         sse =         n = X.shape[0]
#         n_features = X.shape[1]
#         sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
#         se = np.array([
#             np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
#                                                     for i in range(sse.shape[0])
#                     ])
#
#         self.t = self.coef_ / se
#         self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
#         return self
