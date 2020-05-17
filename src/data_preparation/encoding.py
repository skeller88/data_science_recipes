import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_categorical(df, columns):
    dfs_to_concat = [df]
    categories = []
    for column in columns:
        vals = df[column].values.reshape(-1, 1)
        encoder = OneHotEncoder().fit(vals)
        print('encoder categories for', column, encoder.categories_)
        for category in encoder.categories_:
            categories.append(category)
        encoded = encoder.transform(vals).toarray()
        dfs_to_concat.append(pd.DataFrame(encoded, index=df.index, columns=encoder.categories_[0]))

    return pd.concat(dfs_to_concat, axis=1), categories


def get_dummies_drop_level_and_join(df, column, level_to_drop):
    dummies = pd.get_dummies(df[column]).drop([level_to_drop], axis=1)
    return df.join(dummies).drop(column, axis=1)


def categorical_to_ordinal(df):
    converted = []
    converter = {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1
    }

    def convert(row, column):
        return converter[row[column]]

    for column in df.columns:
        if 'TA' in df[column].unique():
            converted.append(column)
            print('converting', column)
            print(df[column].unique())

            def converter_for_column(row):
                return convert(row, column)

            df.loc[:, column] = df.apply(converter_for_column, axis=1)

    return df, converted


def encode_hccf():
    """
    HCCF - high cardinality categorical features.

    "Many studies have shown that One-Hot encoding high cardinality categorical features is not the best way to go,
    especially in tree based algorithms...The basic idea of Target Statistics is simple. We replace a categorical value
    by the mean of all the targets for the training samples with the same categorical value."
    https://deep-and-shallow.com/2020/02/29/the-gradient-boosters-v-catboost/

    TODO - https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html

    :return:
    """
