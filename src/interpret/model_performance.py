import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, log_loss


def get_baselines(y):
    print('random', log_loss(y, np.array([random.uniform(0, 1) for _ in range(len(y))])))
    print('always negative class', log_loss(y, np.array([0 for _ in range(len(y))])))
    print('always positive class', log_loss(y, np.array([1 for _ in range(len(y))])))


def get_precision_recall_df(y_actual, y_pred_probas):
    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred_probas)
    df = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1], 'threshold': thresholds}).sort_values(
        by='threshold')
    return df


def plot_precision_recall_threshold(y_actual, y_pred_probas):
    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred_probas)
    precision = precision[:-1]
    recall = recall[:-1]
    df = pd.concat([
        pd.DataFrame({'stat_name': ['precision' for _ in range(len(precision))],
                      'stat_value': precision,
                      'threshold': thresholds}),
        pd.DataFrame({'stat_name': ['recall' for _ in range(len(precision))],
                      'stat_value': recall,
                      'threshold': thresholds})
    ])

    ax = sns.lineplot(y=df['stat_value'], x=df['threshold'], hue=df['stat_name'],
                      palette={'precision': 'red', 'recall': 'blue'})
    ax.set_title('Precision/Recall per Threshold')
    return ax


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="lightgrey" if cm[i, j] > thresh else "darkgrey")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
