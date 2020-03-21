def binary_distribution(df, column_name):
    num_samples = len(df)
    df['count'] = 1
    dfg = df.groupby(column_name).count()

    positive_class_pct = (dfg.loc[0] / num_samples)[0]
    negative_class_pct = (dfg.loc[1] / num_samples)[0]

    return negative_class_pct, positive_class_pct
