"""
Adopted from http://www.gregreda.com/2015/08/23/cohort-analysis-with-python/
"""

import pandas as pd
import numpy as np


def add_cohort_and_period(df, action_dt_column, action_dt_str) -> pd.DataFrame:
    """
    Assumes index is some sort of user_id to group actions by user.
    :param df:
    :param action_dt_column: example - visit, purchase
    :param action_dt_str: example - '%Y-%m'
    :return:
    """
    # create period columns based on action
    df['period'] = df[action_dt_column].apply(lambda x: x.strftime(action_dt_str))

    # determine the user's cohort group based on their first action
    df['cohort'] = df.groupby(level=0)[action_dt_column].min().apply(lambda x: x.strftime(action_dt_str))

    return df


def group_by_cohort_and_period(df, cohort_column, period_column, user_id_column, action_id_column,
                               action_value_column) -> pd.DataFrame:
    """
    Aggregate unique actions and sum of action values by cohort and period, and create cohort_period
    :param df:
    :param cohort_column: 1, 2, ...
    :param period_column: day, month
    :param user_id_column:
    :param action_id_column: tweet_id, purchase_id
    :param action_value_column: visited, purchase amount
    :return:
    """
    grouped = df.groupby([cohort_column, period_column])

    # count the unique users, orders, and total revenue per Group + Period
    cohorts = grouped.agg({user_id_column: pd.Series.nunique,
                           action_id_column: pd.Series.nunique,
                           action_value_column: np.sum})

    # make the column names more meaningful
    cohorts = cohorts.rename(columns={user_id_column: 'unique_users',
                                      action_id_column: f'unique_{action_id_column}'})

    def cohort_period(df_for_group):
        df_for_group['cohort_period'] = np.arange(len(df_for_group)) + 1
        return df_for_group

    cohorts = cohorts.groupby(level=0).apply(cohort_period)

    return cohorts


def cohorts_with_normalized_actions(cohorts, cohort_group, cohort_period, unique_users):
    """
    Normalize number of actions per cohort group.

    The resulting DataFrame contains the percentage of users from the cohort committing an action within the given
    period.
    :param cohorts:
    :param cohort_group:
    :param cohort_period:
    :param total_users:
    :return:
    """
    cohorts.reset_index(inplace=True)
    cohorts.set_index([cohort_group, cohort_period], inplace=True)

    # create a Series holding the total size of each cohort_group
    cohort_group_size = cohorts[unique_users].groupby(level=0).first()
    return cohorts[unique_users].unstack(0).divide(cohort_group_size, axis=1)


def check_cohort_aggregation():
    pass
    # TODO - implement
    # x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-01')]
    # y = cohorts.ix[('2009-01', '2009-01')]
    #
    # assert(x['UserId'].nunique() == y['TotalUsers'])
    # assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
    # assert(x['OrderId'].nunique() == y['TotalOrders'])