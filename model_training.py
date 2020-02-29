# Try some classifiers with the defaults
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import dump, Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def split(random_state, x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(*[x, y], random_state=random_state, test_size=.2, stratify=y)


def cv_classify_pipelines(random_state, pipeline_output_dir, pipelines, scoring, xtrain, ytrain):
    cv_results = {}
    cv = StratifiedKFold(n_splits=5, random_state=random_state)

    for pipeline_name, pipeline in pipelines.items():
        print('cross validating', pipeline_name)
        # ExtraTreesClassifer with unlimited depth can take up > 2GB of memory
        n_jobs = 2 if pipeline_name == 'extra_trees_classifier' else 6
        cv_results[pipeline_name] = cv_classify(n_jobs=n_jobs, pipeline_name=pipeline_name, pipeline=pipeline, cv=cv,
                                                pipeline_output_dir=pipeline_output_dir,
                                                scoring=scoring, xtrain=xtrain, ytrain=ytrain)

    return cv_results


def cv_classify(n_jobs, pipeline_name, pipeline, cv, pipeline_output_dir, scoring, xtrain, ytrain):
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
        delayed(classify)(**{
            'fold_num': fold_num,
            'pipeline': clone(pipeline),
            'pipeline_name': pipeline_name,
            'pipeline_output_dir': pipeline_output_dir,
            'x': xtrain,
            'y': ytrain,
            'scoring': scoring,
            'train': splits[0],
            'test': splits[1]
        })
        for fold_num, splits in enumerate(cv.split(xtrain, ytrain)))
    return results


def classify(fold_num, pipeline_output_dir, pipeline_name, pipeline, scoring, x, y, train, test):
    """
    Dump pipeline rather than returning it so we don't run out of memory.
    :param fold_num:
    :param pipeline_output_dir:
    :param pipeline_name:
    :param pipeline:
    :param scoring:
    :param x:
    :param y:
    :param train:
    :param test:
    :return:
    """
    results = defaultdict(list)
    start = time.time()
    pipeline_fold = clone(pipeline)
    pipeline_fold.fit(x[train], y[train])
    train_time = time.time() - start
    results['train_time'].append(train_time)
    score = scoring(pipeline_fold, x[test], y[test])
    results['valid_score'].append(score)

    # pipeline_ext = "pipeline" if pipeline_name == "xgb_classifier" else "joblib"
    pipeline_ext = "joblib"
    pipeline_filepath = f'{pipeline_output_dir}/{pipeline_name}_fold_{fold_num}.{pipeline_ext}'
    dump(pipeline_fold, pipeline_filepath)
    return results


def analyze_classification(cv_results):
    flattened_scores = []
    for pipeline_name, score_data in cv_results.items():
        flattened = {key: np.mean(value) for key, value in score_data.items() if key in ['valid_score', 'train_time']}
        flattened['pipeline_name'] = pipeline_name

        flattened_scores.append(flattened)

    return pd.DataFrame(flattened_scores).sort_values(by='valid_score', ascending=False)

# non-default parameters are from https://arxiv.org/pdf/1708.05070.pdf
pipelines = {
    'extra_trees_classifier': Pipeline([
        ('extra_trees_classifier', ExtraTreesClassifier(n_estimators=1000, max_features="log2", criterion="entropy")),
    ]),
    'gradient_boosting_classifier': Pipeline([
        ('gradient_boosting_classifier',
         GradientBoostingClassifier(loss="deviance", learning_rate=.1, n_estimators=500, max_depth=3,
                                    max_features="log2"))
    ]),
    'random_forest_classifier': Pipeline([
        ('random_forest_classifier', RandomForestClassifier(n_estimators=500, max_features=.25, criterion="entropy"))
    ]),
    'knn_classifier': Pipeline([
        ('standard_scaler', StandardScaler()),
        ('knn_classifier', KNeighborsClassifier())
    ]),
    'xgb_classifier': Pipeline([
        ('xgb_classifier', XGBClassifier())
    ])
}

scoring = make_scorer(fbeta_score, beta=2)
pipeline_output_dir = "/home/jovyan/work/data/ultimate_challenge/pipelines"

if not os.path.exists(pipeline_output_dir):
    os.makedirs(pipeline_output_dir, exist_ok=True)

pipeline_cv_results = cv_classify_pipelines(random_state=random_state,
                                            pipeline_output_dir=pipeline_output_dir,
                                            pipelines=pipelines,
                                            scoring=scoring,
                                            xtrain=xtrain,
                                            ytrain=ytrain)


