%matplotlib inline

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


# NLP
import gc
import re
import string
import operator
from collections import defaultdict

import tokenization
from wordcloud import STOPWORDS


def train(model, X_train, y_train, X_valid, y_valid):
    metric_to_monitor = 'val_loss'
    loss = 'mean_squared_error'
    early_stopping_patience = 5
    tolerance = 1e-3
    verbosity = 1

    callbacks = [
        EarlyStopping(monitor=metric_to_monitor, mode='min', min_delta=tolerance,
                      patience=early_stopping_patience, verbose=verbosity, restore_best_weights=True),
    ]
    lr = 1e-3
    optimizer = Adam(learning_rate=lr, epsilon=1e-8)
    n_epochs = 100
    model.compile(loss=loss, optimizer=optimizer)
    return model.fit(x=X_train, y=y_train.values, validation_data=(X_valid, y_valid.values),
                        epochs=n_epochs, batch_size=256,
                        callbacks=callbacks)


def predict(model, X_train, y_train, X_test, y_test, baseline):
    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, predict_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, predict_test))

    r2_train = r2_score(y_train, predict_train)
    r2_test = r2_score(y_test, predict_test)

    if baseline is None:
        baseline = {
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
        }
    print('RMSE train', rmse_train, 'difference from baseline', rmse_train - baseline['rmse_train'])
    print('RMSE test', rmse_test, 'difference from baseline', rmse_test - baseline['rmse_test'])
    print('R2 train', r2_train, 'difference from baseline', r2_train - baseline['r2_train'])
    print('R2 test', r2_test, 'difference from baseline', r2_test - baseline['r2_test'])

    return {
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
    }
