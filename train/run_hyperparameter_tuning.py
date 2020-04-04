
trials = Trials()
best = fmin(score_func_with_defaults, space, algo=tpe.suggest, trials=trials, max_evals=2)