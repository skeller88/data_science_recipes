import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_user_retention(user_retention: pd.DataFrame):
    sns.set(style='white')

    plt.figure(figsize=(15, 8))
    plt.title('Cohorts: User Retention')
    sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%')
