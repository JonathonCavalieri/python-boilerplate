import pandas as pd
import matplotlib.pyplot as plt


def get_stats(data):
    na_count = data.isna().sum().rename("na_count")
    na_perc = ((na_count / len(data)) * 100).rename("na_perc")
    n_unique = data.nunique().rename("nunique")
    stats = data.describe().transpose()
    stats = pd.concat([na_count, na_perc, n_unique, stats], axis=1)
    del na_count, na_perc, n_unique
    return stats


def check_imbalance(data, target_name="target"):
    plt.hist(data[target_name])
    imbalance = data[target_name].sum() / data[target_name].count()
    print(f"{imbalance*100:.0f}% of records are defaults")


def get_target_stats_difference(data, target_name="target"):
    stats_1 = get_stats(data[data[target_name] == 1])
    stats_0 = get_stats(data[data[target_name] == 0])
    stats_diff = stats_1.fillna(0) - stats_0.fillna(0)
    stats_diff_perc = abs(stats_diff / stats_0.fillna(0))

    stats_1.columns = [x + "_1" for x in stats_1.columns]
    stats_0.columns = [x + "_0" for x in stats_0.columns]
    stats_diff.columns = [x + "_diff" for x in stats_diff.columns]
    stats_diff_perc.columns = [x + "_diff_perc" for x in stats_diff_perc.columns]

    stats_combine = pd.concat([stats_1, stats_0, stats_diff, stats_diff_perc], axis=1)

    stats_columns = list(stats_combine.columns)
    stats_columns.sort()
    stats_combine = stats_combine[stats_columns]
    return stats_combine
