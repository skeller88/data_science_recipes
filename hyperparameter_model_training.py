# Does it improve if we hyperparameter tune?
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

from model_training import cv_classify


def score_func(random_state, pipeline_output_dir, n_jobs, pipeline_name, scoring, xtrain, ytrain, params):
    print(params)
    pipeline_output_dir = "/home/jovyan/work/data/ultimate_challenge/pipelines"
    clf = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, random_state=random_state)

    cv_results = cv_classify(n_jobs=n_jobs, pipeline_name=pipeline_name, pipeline=clf, cv=cv,
                             pipeline_output_dir=pipeline_output_dir,
                             scoring=scoring, xtrain=xtrain, ytrain=ytrain)

    return {'loss': np.mean(cv_results['valid_score']), 'status': STATUS_OK}


trials = Trials()
space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 13, 1)),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1)),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'num_class': 9,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'nthread': 6,
    'silent': 1
}

best = fmin(score_func, space, algo=tpe.suggest, trials=trials, max_evals=250)
