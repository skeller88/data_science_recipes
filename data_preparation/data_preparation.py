import pandas as pd


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
