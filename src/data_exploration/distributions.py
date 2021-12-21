from typing import List

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew


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


def boxplot_by_class(x, y, target):
    data_dia = y
    data = x
    data_n_2 = (data - data.mean()) / (data.std())  # standardization
    data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
    data = pd.melt(data, id_vars=target,
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
    # Alternative plots
    # sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
    # sns.violinplot(x="features", y="value", hue=target, data=data, split=True, inner="quart")
    plt.xticks(rotation=90)


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


# skew
def get_skew(df, columns):
    """
    Usage. Skew cutoff from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    skewness = get_skew(df, columns)
    skewed_columns = skewness[skewness['skew'] > 0.75].index
    skewed_columns

    :param df:
    :param columns:
    :return:
    """
    skewed_feats = df[columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'skew': skewed_feats})
    skewness.head(10)
    return skew


def plot_binary_distribution_per_level(df, column_name):
    g = df.groupby(column_name).count().reset_index()
    g.plot(x=column_name, y=g.columns[1], kind='bar')


def class_hists(data, column, target, bins="auto", ax=None, legend=True,
                scale_separately=True):
    """
    Grouped univariate histograms by categorical target.

    From dabl: https://github.com/dabl/dabl/blob/a333bd7/dabl/plot/utils.py#L487

    Parameters
    ----------
    data : pandas DataFrame
        Input data to plot.
    column : column specifier
        Column in the data to compute histograms over (must be continuous).
    target : column specifier
        Target column in data, must be categorical.
    bins : string, int or array-like
        Number of bins, 'auto' or bin edges. Passed to np.histogram_bin_edges.
        We always show at least 5 bins for now.
    ax : matplotlib axes
        Axes to plot into.
    legend : boolean, default=True
        Whether to create a legend.
    scale_separately : boolean, default=True
        Whether to scale each class separately.risk__mock_dental_patients
    Examples
    --------
    >>> class_hists(data, "age", "gender", legend=True)
    <matplotlib...
    """
    col_data = data[column].dropna()

    if ax is None:
        ax = plt.gca()
    if col_data.nunique() > 10:
        ordinal = False
        # histograms
        hist, bin_edges = np.histogram(col_data, bins=bins)
        if len(bin_edges) > 30:
            hist, bin_edges = np.histogram(col_data, bins=30)

        counts = {}
        for name, group in data.groupby(target)[column]:
            this_counts, _ = np.histogram(group, bins=bin_edges)
            counts[name] = this_counts
        counts = pd.DataFrame(counts)
    else:
        ordinal = True
        # ordinal data, count distinct values
        counts = data.groupby(target)[column].value_counts().unstack(target)
    if scale_separately:
        # normalize by maximum
        counts = counts / counts.max()
    bottom = counts.max().max() * 1.1
    for i, name in enumerate(counts.columns):
        if ordinal:
            ax.bar(range(counts.shape[0]), counts[name], width=.9,
                   bottom=bottom * i, tick_label=counts.index, linewidth=2,
                   edgecolor='k', label=name)
            xmin, xmax = 0 - .5, counts.shape[0] - .5
        else:
            ax.bar(bin_edges[:-1], counts[name], bottom=bottom * i, label=name,
                   align='edge', width=(bin_edges[1] - bin_edges[0]) * .9)
            xmin, xmax = bin_edges[0], bin_edges[-1]
        ax.hlines(bottom * i, xmin=xmin, xmax=xmax,
                  linewidth=1)
    if legend:
        ax.legend()
    ax.set_yticks(())
    ax.set_xlabel(column)
    return ax


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
