import math

def check_missing_values(df):
    # Check % missing values
    for column in df.columns:
        missing = df[df[column].isna()]
        if len(missing) > 0:
            print(column, len(missing), len(missing) / len(df) * 100)

def add_dummies(df, columns):
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
