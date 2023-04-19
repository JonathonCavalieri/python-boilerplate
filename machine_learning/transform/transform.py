import numpy as np
import pandas as pd


def cap_outliers(df, whisker_width=1.5):
    df = df.copy()
    for column in df.columns:
        # print(column)

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        limit_lower = q1 - whisker_width * iqr
        limit_upper = q3 + whisker_width * iqr
        print(f"{column} range: {limit_lower} to {limit_upper}")
        outlier_filter = (df[column] <= limit_lower) | (df[column] >= limit_upper)
        moving_average = input[column].shift()
        df.loc[outlier_filter, column] = moving_average[outlier_filter]
        # filter_lower = df[column] <= q1 - whisker_width*iqr
        # filter_upper = df[column] >= q3 + whisker_width*iqr
        # df.loc[filter_lower,column] = q1 - whisker_width*iqr
        # df.loc[filter_upper,column] = q3 + whisker_width*iqr
    return df


def add_cycle_features(input):
    day = 60 * 60 * 24
    year = 365.2425 * day
    input["Seconds"] = input.index.map(pd.Timestamp.timestamp)
    input["Day sin"] = np.sin(input["Seconds"] * (2 * np.pi / day))
    input["Day cos"] = np.cos(input["Seconds"] * (2 * np.pi / day))
    input["Year sin"] = np.sin(input["Seconds"] * (2 * np.pi / year))
    input["Year cos"] = np.cos(input["Seconds"] * (2 * np.pi / year))
    input.drop("Seconds", axis=1, inplace=True)
    return input


# custom aggregate functions
def lm_diff(series):
    if len(series) > 1:
        return series.iloc[-1] - series.iloc[-2]
    else:
        return 0


def squared_mean(series):
    return (series**2).mean()


def missing_values(series):
    return series.isna().sum()


def missing_last_value(series):
    return series.isna().sum()


def missing_last_value(series):
    return series.isna().iloc[-1].astype(int)
