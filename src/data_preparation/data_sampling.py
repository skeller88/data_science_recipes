from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(random_state, x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(*[x, y], random_state=random_state, test_size=.2, stratify=y)
    return xtrain, xtest, ytrain, ytest


def balanced_class_splits(random_state, df, target_class):
    all_negative_class = df[df[target_class] == 0]

    train_negative_class, test_negative_class = train_test_split(
        *[all_negative_class],
        random_state=random_state,
        test_size=.2)

    all_positive_class = df[df[target_class] == 1]
    train_positive_class = all_positive_class.sample(
        n=len(train_negative_class), random_state=random_state)

    test_positive_class = all_positive_class[~all_positive_class.inde.isin(train_positive_class.inde)]

    train = pd.concat([train_positive_class, train_negative_class])
    test = pd.concat([test_positive_class, test_negative_class])

    assert len(train_positive_class) == len(train_negative_class)
    assert len(all_negative_class) == len(test_negative_class) + len(train_negative_class)
    assert len(all_positive_class) == len(test_positive_class) + len(train_positive_class)

    return train, test

def binned_splits(random_state, df, target_class):
    """
    Split continuous data.

    https://bookdown.org/max/FES/data-splitting.html

    https://github.com/scikit-learn/scikit-learn/issues/4757
    :param random_state:
    :param df:
    :param target_class:
    :return:
    """
    bins = np.linspace(0, 50, 25)
    y = df[target_class].values
    y_binned = np.digitize(y, bins)
    xtrain, xtest, ytrain, ytest = train_test_split(df.drop(target_class, axis=1), y, stratify=y_binned,
                                                        random_state=random_state)
    return xtrain, xtest, ytrain, ytest


def validate_splits(xtrain, xtest, ytrain, ytest):
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    assert xtrain.shape[1] == xtest.shape[1]
    assert len(ytrain.shape) == len(ytest.shape)

    assert xtrain.shape[0] == ytrain.shape[0]
    assert xtest.shape[0] == ytest.shape[0]


def train_valid_test_splits(*dataframes: List[pd.DataFrame]):
    train_dfs = []
    valid_dfs = []
    test_dfs = []

    train_size = .8
    for dataframe in dataframes:
        random_mask = np.random.rand(len(dataframe))
        train_mask = random_mask < train_size
        valid_mask = (train_size <= random_mask) & (random_mask < .9)
        test_mask = random_mask >= .9

        train_df = dataframe[train_mask]
        valid_df = dataframe[valid_mask]
        test_df = dataframe[test_mask]

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)

        assert len(dataframe) == len(train_df) + len(test_df) + len(valid_df)
        for df in [train_df, valid_df, test_df]:
            assert len(df[df.duplicated()]) == 0

    train = pd.concat(train_dfs)
    valid = pd.concat(valid_dfs)
    test = pd.concat(test_dfs)

    for df in [train, valid, test]:
        assert len(df[df.duplicated()]) == 0

    print("len(train)", len(train), "len(valid)", len(valid), "len(test)", len(test))
    assert len(train) + len(valid) + len(test) == sum(map(len, dataframes))
    return train, valid, test
