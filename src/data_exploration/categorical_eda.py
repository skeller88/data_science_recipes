def get_categorical_counts(df, column):
    return df.groupby(column).count()[df.columns[0]].sort_values(ascending=False)


def positive_pct_per_level(df, column, target):
    """
    Return count and % of positive class at all levels of feature.

    Example:
    col = 'device'
    stats = positive_pct_per_level(df, col, 'converted')
    print(stats)
    stats.plot(x=col, y='converted_pct', kind='bar', figsize=(15, 4))
    stats.plot(x=col, y='frequency_pct', kind='bar', figsize=(15, 4))

    :param df:
    :param column:
    :return:
    """
    g = df[[column, target]].groupby(column).agg(['sum', 'count']).reset_index()
    target_pct = f"{target}_pct"
    g[target_pct] = g[target]['sum'] / g[target]['count']
    g['frequency_pct'] = g[target]['count'] / len(df)
    return g.sort_values(by=target_pct, ascending=False)
