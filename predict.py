from joblib import load

import numpy as np


def kfold_predict_class(x, model_path_prefix, n_folds, class_threshold):
    all_predicted_probas = []
    for fold_num in n_folds:
        model = load(f"{model_path_prefix}_fold{fold_num}.joblib")
        predicted_probas = model.predict_proba(x)
        all_predicted_probas.append(np.array(predicted_probas))

    mean_predicted_probas = np.stack(all_predicted_probas).mean(axis=0)
    predicted_classes = np.array([0 if proba < class_threshold else 1 for proba in mean_predicted_probas ])

    return predicted_classes, mean_predicted_probas
