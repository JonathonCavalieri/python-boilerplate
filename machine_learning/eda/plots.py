import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def plot_hist(
    train,
    columns,
    target_label,
    width=4,
    cap_outliers=False,
    bounds=[0.01, 0.99],
    nbins=10,
    ax_size=(4, 4),
):
    height = ceil(len(columns) / width)

    columns = np.array(columns)
    columns.resize(width * height)

    columns = columns.reshape((-1, width))

    if cap_outliers:
        train = train.copy()

    ax_height, ax_width = ax_size

    fig = plt.figure(
        constrained_layout=True, figsize=(ax_width * width, ax_height * height)
    )
    ax = fig.subplots(nrows=height, ncols=width)

    for y, row in enumerate(columns):
        for x, col in enumerate(row):
            if col == "":
                continue

            column_temp = train[[col, target_label]].copy()

            if cap_outliers:
                bounds = column_temp[col].quantile([0.01, 0.99]).values
                lower = bounds[0]
                upper = bounds[1]
                print(lower, upper)
                column_temp[col] = column_temp[col].clip(lower, upper)

            if height == 1:
                coord = x
            else:
                coord = (y, x)

            ax[coord].set_title(col)
            _, bins, _ = ax[coord].hist(
                column_temp[col], alpha=0.5, color="Grey", bins=nbins
            )
            ax[coord].hist(
                column_temp[column_temp[target_label] == 0][col],
                label="0",
                alpha=0.5,
                color="Blue",
                bins=bins,
            )
            ax[coord].hist(
                column_temp[column_temp[target_label] == 1][col],
                label="1",
                alpha=1,
                color="Orange",
                bins=bins,
            )
