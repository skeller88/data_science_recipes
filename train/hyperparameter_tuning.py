import json

import pandas as pd
from hyperopt import STATUS_OK
from sklearn.model_selection import StratifiedKFold

from train.model_training import cv_classify


def score_func(random_state, pipeline_factory, pipeline_output_dir, n_jobs, pipeline_name, scoring_func, xtrain, ytrain, params):
    param_hash = hash(json.dumps(params))
    pipeline = pipeline_factory(**params)
    cv = StratifiedKFold(n_splits=5, random_state=random_state)
    pipeline_name_for_experiment = f"{pipeline_name}_{param_hash}"
    cv_results = cv_classify(n_jobs=n_jobs, pipeline_name=pipeline_name_for_experiment, pipeline=pipeline, cv=cv,
                             pipeline_output_dir=pipeline_output_dir,
                             scoring_func=scoring_func, xtrain=xtrain, ytrain=ytrain)

    results = pd.DataFrame(cv_results).mean().to_dict()
    results['pipeline_name'] = pipeline_name_for_experiment
    valid_loss = pd.DataFrame(cv_results)['valid_loss'].mean()

    for param_name, param_value in params.items():
        results[param_name] = param_value

    return {'loss': valid_loss, 'status': STATUS_OK, 'results': results}


def analyze_trials(trials):
    flattened = []
    for trial in trials.trials:
        result = trial['result']
        trial_dict = result['results']
        trial_dict['loss'] = result['loss']
        flattened.append(trial_dict)

    results = pd.DataFrame(flattened)
    return results.sort_values(by='loss')
