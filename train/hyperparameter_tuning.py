import json

import pandas as pd
from hyperopt import STATUS_OK
from sklearn.model_selection import StratifiedKFold

from train.model_training import cv_classify


def score_func(random_state, pipeline_factory, pipeline_output_dir, n_jobs, pipeline_name, scoring, xtrain, ytrain, params):
    param_hash = hash(json.dumps(params))
    pipeline = pipeline_factory(**params)
    cv = StratifiedKFold(n_splits=5, random_state=random_state)
    pipeline_name_for_experiment = f"{pipeline_name}_{param_hash}"
    cv_results = cv_classify(n_jobs=n_jobs, pipeline_name=pipeline_name_for_experiment, pipeline=pipeline, cv=cv,
                             pipeline_output_dir=pipeline_output_dir,
                             scoring=scoring, xtrain=xtrain, ytrain=ytrain)

    means = pd.DataFrame(cv_results).mean().to_dict()
    means['pipeline_name'] = pipeline_name_for_experiment
    valid_score = pd.DataFrame(cv_results)['valid_score'].mean()
    return {'loss': 1 - valid_score, 'status': STATUS_OK, 'results': means}



# trials = Trials()
# best = fmin(score_func, space, algo=tpe.suggest, trials=trials, max_evals=250)