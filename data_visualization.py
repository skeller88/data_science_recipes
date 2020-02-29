def histogram_grid(df, columns):
    dim = math.ceil(len(columns) / 2)
    f, axes = plt.subplots(dim, dim, figsize=(20, 20))
    for ax, feature in zip(axes.flat, columns):
        sns.distplot(df[feature], color="skyblue", ax=ax)
