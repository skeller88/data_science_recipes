from joblib import load


def get_xgb_feature_importance(actual_feature_names, model_path):
    clf = load(model_path)
    features = clf._final_estimator.get_booster().get_score(importance_type="gain")
    actual_features = []
    for feature_name, gain in features.items():
        idx = int(feature_name.strip('f'))
        actual_feature_name = actual_feature_names[idx]
        actual_features.append({'feature_name': actual_feature_name, 'gain': gain})

    return pd.DataFrame(actual_features).sort_values(by='gain', ascending=False)