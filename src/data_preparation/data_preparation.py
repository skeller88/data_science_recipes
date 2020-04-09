import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import FunctionTransformer

from src.data_exploration.eda import get_types

import numpy as np


def make_pipeline(df):
    x = df
    col_dtypes = get_types(x)

    encoder = ColumnTransformer(
        [('categorical', CatBoostEncoder(), col_dtypes['object']),
         # could use passthrough=remainder, but this way makes column ordering more obvious
         ('numeric', FunctionTransformer(), col_dtypes['int64'] + col_dtypes['float64'])
         ]
    )

    all_columns_idx = np.full((len(x)), True, dtype=bool)
    imputer = ColumnTransformer(
        [('knn_imputer', KNNImputer(), all_columns_idx)]
    )

    pipeline = Pipeline(steps=[
        ('encoder', encoder),
        ('imputer', imputer),
    ])

    return pipeline, col_dtypes['object'] + col_dtypes['int64'] + col_dtypes['float64']


def check_duplicates(df):
    for column in df.columns:
        duplicated = df[df[column].duplicated()]
        if len(duplicated) > 0:
            print(column, len(duplicated), len(duplicated) / len(df) * 100)


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


def find_mislabeled(df):
    df_mislabeled = df.groupby(['text']).nunique().sort_values(by='target', ascending=False)
    df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
    df_mislabeled.index.tolist()
