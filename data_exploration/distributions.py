from typing import List

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def print_relative_frequencies(df, columns):
    for column in columns:
        print('\n')
        print(column, df[column].value_counts() / len(df) * 100)


def histogram_grid(df, columns, should_plot_missing):
    """
    :param df:
    :param columns
    :return:
    """
    if len(columns) < 5:
        n_cols = 2
    else:
        n_cols = 5
    n_rows = math.ceil(len(columns) / n_cols)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    for ax, feature in zip(axes.flat, sorted(columns)):
        data = df[feature]
        if should_plot_missing:
            data = data
        data.hist(ax=ax)
        ax.set_title(feature, pad=-10)


def boxplot(df, columns):
    df[columns].plot.box(vert=False, figsize=(15, 10))


def distplot_grid(df, columns):
    """
    Usage if there are lots of parameters

    for start in range(0, len(col_dtypes['int64']), 16):
        distplot_grid(df, col_dtypes['int64'][start:start+16])
    :param df:
    :param columns: Must be numeric type
    :return:
    """
    if len(columns) < 5:
        n_cols = 2
    else:
        n_cols = 4

    n_rows = math.ceil(len(columns) / n_cols)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    for ax, feature in zip(axes.flat, columns):
        try:
            sns.distplot(df[feature], hist=False, color="skyblue", ax=ax)
        except Exception as ex:
            print('Could not set KDE for feature', feature)
            df[feature].hist(ax=ax)
        ax.set_title(feature, pad=-10)


def binary_distribution(df, column_name):
    num_samples = len(df)
    df['count'] = 1
    dfg = df.groupby(column_name).count()

    positive_class_pct = (dfg.loc[0] / num_samples)[0]
    negative_class_pct = (dfg.loc[1] / num_samples)[0]

    return negative_class_pct, positive_class_pct


def plot_variable_dists_by_class(df, target_column: str, features: List[str]):
    has_target: pd.Series = df[target_column] == 1

    fig, axes = plt.subplots(ncols=1, nrows=len(features), figsize=(20, 50), dpi=100)

    for i, feature in enumerate(features):
        sns.distplot(df.loc[~has_target][feature], label=f'not {target_column}', ax=axes[i], color='green')
        sns.distplot(df.loc[has_target][feature], label=f'{target_column}', ax=axes[i], color='red')

        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=12)
        axes[i].tick_params(axis='y', labelsize=12)
        axes[i].legend()

        axes[i].set_title(f'{feature} Target Distribution', fontsize=13)
    plt.show()


def plot_variable_dists_by_class_and_dataset(df_train, df_test, target_column: str, features: List[str]):
    has_target: pd.Series = df_train[target_column] == 1

    fig, axes = plt.subplots(ncols=2, nrows=len(features), figsize=(20, 50), dpi=100)

    for i, feature in enumerate(features):
        sns.distplot(df_train.loc[~has_target][feature], label=f'Not {target_column}', ax=axes[i][0], color='green')
        sns.distplot(df_train.loc[has_target][feature], label=f'{target_column}', ax=axes[i][0], color='red')

        sns.distplot(df_train[feature], label='Training', ax=axes[i][1])
        sns.distplot(df_test[feature], label='Test', ax=axes[i][1])

        for j in range(2):
            axes[i][j].set_xlabel('')
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].legend()

        axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
        axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

    plt.show()
