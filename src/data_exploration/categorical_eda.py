def positive_pct_per_level(df, feature, target):
    """
    Return count and % of level that belongs to the positive class at all
    levels of feature.

    Example:
    feature = 'device'
    ppl = positive_pct_per_level(df, feature, 'converted')
    ppl[(ppl[target] == True) & (ppl['count'] > 5)].sort_values(
    by='pct_positive',
    ascending=False)
    fig, ax = plt.subplots(figsize=(15, 4))
    ppl.set_index(feature)[['pct_positive', 'count']][:20].plot(kind='bar',
    secondary_y=['count'], ax=ax)

    :param df:
    :param feature:
    :return:
    """
    # fill in levels with 0 values so that they aren't dropped
    # https://stackoverflow.com/questions/37003100/pandas-groupby-for-zero-values
    counts = df.groupby([target, feature]).count()[['count']].unstack(
        fill_value=0).stack()
    counts['pct_positive'] = counts.groupby(level=feature).apply(
        lambda x: 100 * x / float(x.sum()))
    counts['pct_total'] = counts['count'] / counts['count'].sum() * 100

    # TODO - add pct_total_positives
    return counts[['pct_total', 'pct_positive', 'count']].sort_values(
        by='pct_positive',
        ascending=False)
