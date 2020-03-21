from joblib import load

import numpy as np
import pandas as pd


def kfold_predict_class(x, model_path_prefix, n_folds, class_threshold):
    all_predicted_probas = []
    for fold_num in n_folds:
        model = load(f"{model_path_prefix}_fold{fold_num}.joblib")
        predicted_probas = model.predict_proba(x)[:, 1]
        all_predicted_probas.append(np.array(predicted_probas))

    mean_predicted_probas = np.stack(all_predicted_probas).mean(axis=0)
    predicted_classes = np.array([0 if proba < class_threshold else 1 for proba in mean_predicted_probas ])

    return predicted_classes, mean_predicted_probas


def predict_class(x, model, class_threshold):
    predicted_probas = model.predict(x)[:, 1]
    predicted_classes = np.array([0 if proba < class_threshold else 1 for proba in predicted_probas])

    return predicted_classes, predicted_probas


def analyze_classification(pipeline_cv_results):
    mean_scores = []
    for pipeline_name, score_data in pipeline_cv_results.items():
        means = pd.DataFrame(pipeline_cv_results[pipeline_name]).mean().to_dict()
        means['pipeline_name'] = pipeline_name

        mean_scores.append(means)

    return pd.DataFrame(mean_scores).sort_values(by='valid_loss', ascending=False)
