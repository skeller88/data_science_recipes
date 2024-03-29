import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_types(df):
    dtypes = defaultdict(list)
    for col, dtype in df.dtypes.items():
        dtypes[str(dtype)].append(col)
    return dtypes


def groupby_count(df, column):
    if 'count' not in df.columns:
        return 'df must have a "count" column for this method to work.'
    return df.groupby(column).count()['count'].sort_values(ascending=False)


def figure_and_axes(columns, n_cols=None):
    if n_cols is None:
        if len(columns) < 5:
            n_cols = 2
        else:
            n_cols = 4

    n_rows = math.ceil(len(columns) / n_cols)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    return f, axes


def plot_scatter_against_target(df, columns, target):
    if len(columns) < 5:
        n_cols = 2
    else:
        n_cols = 4

    n_rows = math.ceil(len(columns) / n_cols)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    for ax, feature in zip(axes.flat, columns):
        sns.scatterplot(x=df[feature], y=df[target], color="skyblue", ax=ax)
        ax.set_title(feature, pad=-10)


def plot_boxplot_against_target(df, feature, target):
    f, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(x=feature, y=target, data=df[[feature, target]], ax=ax)
    plt.xticks(rotation=90)


def pairplot(df, cols):
    sns.pairplot(df[cols], height=2.5)


def plot_multiple_plots(max_features_per_plot, df, columns, plot_func, **plot_func_args):
    """
    Example:

    plot_multiple_plots(12, df, df.columns, plot_scatter_against_target, **{'target':target})

    :param max_features_per_plot:
    :param df:
    :param columns:
    :param plot_func:
    :param plot_func_args:
    :return:
    """
    num_plots = math.ceil(len(columns) / max_features_per_plot)
    features_per_plot = math.ceil(len(columns) / num_plots)
    for plot_num in range(num_plots):
        start = features_per_plot * plot_num
        end = start + features_per_plot
        plot_func(df, columns[start:end], **plot_func_args)


# Correlation and VIF

def correlation_heatmap(df, target=None, n_largest=None):
    f, ax = plt.subplots(figsize=(12, 9))
    # evidence for using both Kendall's Tau and Spearman's rho if the feature distributions are not
    # assumed to be normal: https://stats.stackexchange.com/questions/3943/kendall-tau-or-spearmans-rho
    # Some commenters say that Spearman's rho is more intepretable because it extends the idea of R^
    # in that it quantifies the difference between the % of concordant and discordant pairs among all
    # possible pairwise events.
    corrmat = df.corr(method='spearman')
    if n_largest is not None:
        cols = corrmat.nlargest(n_largest, target)  [target].index
        cm = np.corrcoef(df[cols].values.T)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                    xticklabels=cols.values)
    else:
        cols = []
        sns.heatmap(corrmat, vmax=.8, square=True)

    return cols


def abs_correlation_heatmap(df, target=None):
    f, ax = plt.subplots(figsize=(12, 9))
    corrmat = df.corr(method='spearman')
    corr_abs = corrmat.abs() > 0.5
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_abs, annot=True)


def get_vif(x):
    x_matrix = x.as_matrix()
    vif = [variance_inflation_factor(x_matrix, i) for i in range(x_matrix.shape[1])]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = x.columns
    vif_factors['vif'] = vif
    return vif_factors.sort_values(by=['vif'], ascending=False)

