trials = Trials()
best = fmin(score_func_with_defaults, xgb_classifier_space, algo=tpe.suggest, trials=trials, max_evals=2)