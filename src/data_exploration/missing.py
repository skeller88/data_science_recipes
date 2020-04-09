from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_missing_stats(df, columns):
    dtypes = df.dtypes
    missing_cols = []
    for column in columns:
        missing = df[df[column].isnull()]

        if len(missing) > 0:
            missing_cols.append({'column': column,
                                 'dtype': str(dtypes[column]),
                                 'num_missing': len(missing),
                                 'num_not_missing': len(df) - len(missing),
                                 'pct_missing': len(missing) / len(df)})

    return pd.DataFrame(missing_cols).sort_values(by='pct_missing', ascending=False)


def plot_missing_variable_count(df_train, df_test, missing_cols: List[str]):
    """
    Compare missing  values of two dataframes
    :param df_train:
    :param df_test:
    :return:
    """
    fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

    sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values,
                ax=axes[0])
    sns.barplot(x=df_test[missing_cols].isnull().sum().index, y=df_test[missing_cols].isnull().sum().values, ax=axes[1])

    axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)

    axes[0].set_title('Training Set', fontsize=13)
    axes[1].set_title('Test Set', fontsize=13)

    plt.show()
