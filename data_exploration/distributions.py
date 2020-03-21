from typing import List

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def histogram_grid(df, columns):
    """
    :param df:
    :param columns: Must be numeric type
    :return:
    """
    dim = math.ceil(math.sqrt(len(columns)))
    f, axes = plt.subplots(dim, dim, figsize=(20, 20))
    for ax, feature in zip(axes.flat, columns):
        sns.distplot(df[feature], color="skyblue", ax=ax)


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
