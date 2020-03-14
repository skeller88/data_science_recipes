from joblib import load

import pandas as pd
import numpy as np
from rfpimp import importances, plot_importances
import shap
from sklearn.metrics import fbeta_score


def get_xgb_feature_importance(actual_feature_names, model_path):
    clf = load(model_path)
    features = clf._final_estimator.get_booster().get_score(importance_type="gain")
    actual_features = []
    for feature_name, gain in features.items():
        idx = int(feature_name.strip('f'))
        actual_feature_name = actual_feature_names[idx]
        actual_features.append({'feature_name': actual_feature_name, 'gain': gain})

    return pd.DataFrame(actual_features).sort_values(by='gain', ascending=False)


def get_xgb_shap(pipeline_results, xtrain: np.array, feature_names):
    """
    Use SHAP: https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    :param pipeline_results:
    :param xtrain:
    :param feature_names:
    :return:
    """
    explainer = shap.TreeExplainer(pipeline_results._final_estimator)
    shap_values = explainer.shap_values(xtrain[:1000])
    shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values[0,:], xtrain[0,:], feature_names=xdf.columns)
    return pd.DataFrame({'shap_value': shap_values[0, :], 'feature_names': feature_names}).sort_values(by='shap_value',
                                                                                                ascending=False)

def show_rf_feature_importance(clf, x: pd.DataFrame, y: pd.DataFrame):
    def fbeta2(clf, x, y):
        return fbeta_score(y, clf.predict(x), beta=2)

    importances = importances(clf, x, y, fbeta2)
    viz = plot_importances(importances)
    viz.view()

