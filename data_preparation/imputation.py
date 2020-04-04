from sklearn.impute import SimpleImputer
import pandas as pd




def simple_impute(df, col_dtypes, categorical_strategy='most_frequent', numerical_strategy='mean'):
    imputed_categorical = pd.DataFrame(
        SimpleImputer(strategy=categorical_strategy).fit_transform(df[col_dtypes['object']]),
        columns=col_dtypes['object']
    )

    imputed_int = pd.DataFrame(
        SimpleImputer(strategy=numerical_strategy).fit_transform(df[col_dtypes['int64']]),
        columns=col_dtypes['int64']
    )

    imputed_float = pd.DataFrame(
        SimpleImputer(strategy=numerical_strategy).fit_transform(df[col_dtypes['float64']]),
        columns=col_dtypes['float64']
    )

    df_imputed = pd.concat([imputed_categorical, imputed_int, imputed_float], axis=1)
    assert len(df_imputed) == len(df)
    assert len(df_imputed.columns) == len(df.columns)

    return df_imputed
