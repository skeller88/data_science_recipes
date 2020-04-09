# Try some regressors with the defaults
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, ElasticNet
from sklearn.metrics import fbeta_score, make_scorer, \
    log_loss, mean_squared_error
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data_preparation.binned_stratified_kfold import BinnedStratifiedKFold


def pipeline_regressions(random_state, transformer_components: List[Tuple],
                         all_estimator_pipeline_components: Dict[str, List[Tuple]], search_space, score_funcs, x, y,
                         n_bins):
    """

    :param random_state:
    :param transformer_components:
    :param all_estimator_pipeline_components:
    :param search_space:
    :param score_funcs:
    :param x:
    :param y:
    :param n_bins:
    :return:
    """
    results = {}
    cv = BinnedStratifiedKFold(random_state=random_state, shuffle=True, n_bins=n_bins)
    hp_cv = BinnedStratifiedKFold(random_state=random_state, shuffle=True, n_bins=n_bins)
    for estimator_name, estimator_pipeline_components in all_estimator_pipeline_components.items():
        print('training', estimator_name)

        pipeline: Pipeline = Pipeline(steps=transformer_components + estimator_pipeline_components)

        if search_space is not None:
            pipeline = RandomizedSearchCV(
                estimator=pipeline, param_distributions=search_space[estimator_name], scoring=score_funcs, cv=hp_cv)
        results[estimator_name] = cross_validate(pipeline, x, y, scoring=score_funcs, cv=cv)

    return results


def analyze_results(pipeline_results):
    results = [result for result in pipeline_results.values()]
    return pd.DataFrame(results).sort_values(by='valid_loss')


def analyze_cv_classification(pipeline_results):
    mean_scores = []
    for pipeline_name in pipeline_results.keys():
        means = {key: np.mean(value) for key, value in pipeline_results[pipeline_name].items()}
        means['pipeline_name'] = pipeline_name
        mean_scores.append(means)

    return pd.DataFrame(mean_scores)


# non-default parameters are from https://arxiv.org/pdf/1708.05070.pdf
estimators = {
    'extra_trees_regressor': [
        ('extra_trees_regressor', ExtraTreesRegressor()),
    ],
    'gradient_boosting_regressor': [
        ('gradient_boosting_regressor', GradientBoostingRegressor())
    ],
    'random_forest_regressor': [
        ('random_forest_regressor', RandomForestRegressor())
    ],
    'knn_regressor': [
        ('standard_scaler', StandardScaler()),
        ('knn_regressor', KNeighborsRegressor())
    ],
    'xgb_regressor': [
        ('xgb_regressor', XGBRegressor())
    ],
    'lasso_regressor': [
        ('standard_scaler', StandardScaler()),
        ('lasso_regressor', Lasso())
    ],
    'ridge_regressor': [
        ('ridge_regressor', Ridge())
    ],
    'elastic_net_regressor': [
        ('elastic_net_regressor', ElasticNet())
    ],
    'sgd_regressor': [
        ('sgd_regressor', SGDRegressor())
    ],
}


def main(random_state, xtrain, ytrain, pipeline_output_dir):
    score_funcs = {
        'fbeta2': make_scorer(fbeta_score, beta=2)
    }

    if not os.path.exists(pipeline_output_dir):
        os.makedirs(pipeline_output_dir, exist_ok=True)

    return pipeline_regressions(random_state=random_state,
                                pipeline_output_dir=pipeline_output_dir,
                                pipelines=pipelines,
                                loss_func=log_loss,
                                score_funcs=score_funcs,
                                xtrain=xtrain,
                                ytrain=ytrain)
