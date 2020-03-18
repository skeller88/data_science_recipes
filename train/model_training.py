# Try some classifiers with the defaults
import os
import time
from typing import Dict, Callable

import numpy as np
import pandas as pd
from joblib import dump, Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer, \
accuracy_score, average_precision_score, confusion_matrix, log_loss, precision_recall_fscore_support, \
precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def split(random_state, x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(*[x, y], random_state=random_state, test_size=.2, stratify=y)
    return xtrain, xtest, ytrain, ytest


def classify_pipelines(random_state, pipeline_output_dir, pipelines, loss_func, score_funcs, xtrain, ytrain):
    results = {}
    idx = np.array([num for num in range(len(xtrain))])
    train_idx, valid_idx = train_test_split(idx, random_state=random_state)
    for pipeline_name, pipeline in pipelines.items():
        print('training', pipeline_name)
        results[pipeline_name] = classify(fold_num=0, pipeline_name=pipeline_name, pipeline=pipeline,
                                          pipeline_output_dir=pipeline_output_dir,
                                          loss_func=loss_func,
                                          score_funcs=score_funcs, x=xtrain, y=ytrain, train=train_idx,
                                          test=valid_idx)

    return results


def cv_classify_pipelines(random_state, pipeline_output_dir, pipelines, loss_func, score_funcs, xtrain, ytrain):
    cv_results = {}
    cv = StratifiedKFold(n_splits=5, random_state=random_state)

    for pipeline_name, pipeline in pipelines.items():
        print('training', pipeline_name)
        # ExtraTreesClassifer with unlimited depth can take up > 2GB of memory
        n_jobs = 2 if pipeline_name == 'extra_trees_classifier' else 6
        cv_results[pipeline_name] = cv_classify(n_jobs=n_jobs, pipeline_name=pipeline_name, pipeline=pipeline, cv=cv,
                                                pipeline_output_dir=pipeline_output_dir,
                                                loss_func=loss_func,
                                                score_funcs=score_funcs, xtrain=xtrain, ytrain=ytrain)

    return cv_results


def cv_classify(n_jobs, pipeline_name, pipeline, cv, pipeline_output_dir, loss_func, score_funcs, xtrain, ytrain):
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
        delayed(classify)(**{
            'fold_num': fold_num,
            'pipeline': clone(pipeline),
            'pipeline_name': pipeline_name,
            'pipeline_output_dir': pipeline_output_dir,
            'x': xtrain,
            'y': ytrain,
            'loss_func': loss_func,
            'score_funcs': score_funcs,
            'train': splits[0],
            'test': splits[1]
        })
        for fold_num, splits in enumerate(cv.split(xtrain, ytrain)))
    return results


def classify(fold_num, pipeline_output_dir, pipeline_name, pipeline, loss_func, score_funcs: Dict[str, Callable], x, y, train, test):
    """
    Dump pipeline rather than returning it so we don't run out of memory.
    :param fold_num:
    :param pipeline_output_dir:
    :param pipeline_name:
    :param pipeline:
    :param score_funcs:
    :param x:
    :param y:
    :param train:
    :param test:
    :return:
    """
    results = {}
    start = time.time()
    pipeline_fold = clone(pipeline)
    pipeline_fold.fit(x[train], y[train])
    train_time = time.time() - start
    results['train_time'] = train_time
    pred_probas = pipeline_fold.predict_proba(x[test])[:, 1]
    preds = np.array([0 if prob < .5 else 1 for prob in pred_probas])
    results['valid_loss'] = loss_func(y[test], pred_probas)

    for score_func_name, score_func in score_funcs.items():
        results[f'valid_{score_func_name}'] = score_func(y[test], preds)

    # pipeline_ext = "pipeline" if pipeline_name == "xgb_classifier" else "joblib"
    pipeline_ext = "joblib"
    pipeline_filepath = f'{pipeline_output_dir}/{pipeline_name}_fold_{fold_num}.{pipeline_ext}'
    results['pipeline_filepath'] = pipeline_filepath
    dump(pipeline_fold, pipeline_filepath)
    return results


def analyze_classification(pipeline_results):
    mean_scores = []
    for pipeline_name, score_data in pipeline_results.items():
        index = [num for num in range(len(score_data))]
        means = pd.DataFrame(pipeline_results[pipeline_name], index=index).mean().to_dict()
        means['pipeline_name'] = pipeline_name
        mean_scores.append(means)

    return pd.DataFrame(mean_scores).sort_values(by='valid_loss')


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


def main(xtrain, ytrain):
    random_state = 42
    score_funcs = {
        'fbeta2': make_scorer(fbeta_score, beta=2)
    }
    pipeline_output_dir = "/home/jovyan/work/data/ultimate_challenge/pipelines"

    if not os.path.exists(pipeline_output_dir):
        os.makedirs(pipeline_output_dir, exist_ok=True)

    pipeline_results = classify_pipelines(random_state=random_state,
                                                pipeline_output_dir=pipeline_output_dir,
                                                pipelines=pipelines,
                                                loss_func=log_loss,
                                                score_funcs=score_funcs,
                                                xtrain=xtrain,
                                                ytrain=ytrain)
