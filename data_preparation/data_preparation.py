import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def check_duplicates(df):
    for column in df.columns:
        duplicated = df[df[column].duplicated()]
        if len(duplicated) > 0:
            print(column, len(duplicated), len(duplicated) / len(df) * 100)


def check_missing_values(df):
    # Check % missing values
    for column in df.columns:
        missing = df[df[column].isna()]
        if len(missing) > 0:
            print(column, len(missing), len(missing) / len(df) * 100)


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


def add_decomposed_date_variables(df, columns, date_parts=['year', 'month', 'day']):
    # Decompose date variables
    date_columns = []
    for date_column in columns:
        datetime_column = pd.to_datetime(df[date_column]).dt
        for date_part in date_parts:
            date_column = f"{date_column}_{date_part}"
            date_columns.append(date_column)
            df[date_column] = getattr(datetime_column, date_part)

    return df, date_columns


def impute_missing(df, missing_columns):
    # Impute missing variables. Assume MCAR
    for column in missing_columns:
        print(column, 'median', df[column].median())
        missing = df[df[column].isna()]
        df.loc[missing.index, column] = df[column].median()
    return df

def find_mislabeled(df):
    df_mislabeled = df.groupby(['text']).nunique().sort_values(by='target', ascending=False)
    df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
    df_mislabeled.index.tolist()
