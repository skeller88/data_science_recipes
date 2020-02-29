from hyperopt import hp
from hyperopt.pyll.base import scope
from xgboost import XGBClassifier

from train.hyperparameter_tuning import score_func

# Haven't found a paper yet that gives the best parameters
xgb_classifier_space = {
    # mlm - 50 - 400
    # n_estimators
    # av - 3 - 10
    # # kaggle tilii7 - 3-5
    # tds - hp tuning the smart way - 3 - 15
    'max_depth': scope.int(hp.quniform('max_depth', 1, 9, 1)),
    # av - .01 - .2
    # tds - hp tuning the smart way - .05 - .3
    'learning_rate': hp.quniform('eta', 0.025, 0.5, 0.025),
    # 'learning_rate': hp.quniform('eta', 0.025, 0.5, 0.025),
    # Don't tune the booster. AnalyticsVidhya says gblinear rarely used, and
    # XGB documentation says dart is slow.
    # 'booster': hp.choice(['gbtree', 'dart']),
    # Already parallelzing cv training
    'n_jobs': 1,
    # kaggle tilii7 - .5 - 5 with .5 step
    # tds - hp tuning the smart way - 0 - .4 with .1 step
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    # kaggle tilii7 - 1-10
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1)),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    # tds - hp tuning the smart way - .3 - .7
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
}


def xgb_pipeline_factory(**params):
    """
    From xgboost documentation: https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=n_estimators
    "**kwargs is unsupported by scikit-learn. We do not guarantee that parameters passed via this argument will
    interact properly with scikit-learn."
    :param params:
    :return:
    """
    return XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        gamma=params['gamma'],
        colsample_bytree=params['colsample_bytree'],
    )


def score_func_with_defaults(params):
    pipeline_output_dir = "/home/jovyan/work/data/ultimate_challenge/pipelines"
    return score_func(random_state=random_state,
                      pipeline_factory=xgb_pipeline_factory,
                      pipeline_output_dir=pipeline_output_dir,
                      n_jobs=n_jobs,
                      pipeline_name=pipeline_name,
                      scoring=scoring,
                      xtrain=xtrain,
                      ytrain=ytrain,
                      params=params)
