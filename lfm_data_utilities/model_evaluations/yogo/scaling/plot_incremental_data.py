#! /usr/bin/env python3

import polars as pl
import matplotlib.pyplot as plt

pl.Config.set_tbl_rows(100)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="wandb csv export")
    args = parser.parse_args()

    q = pl.scan_csv(args.csv)

    cols = (
        "training set size",
        "Name",
        "Notes",
        "best_val_loss",
        "test loss",
        "test mAP",
        "test precision",
        "test recall",
    )

    numerical_cols = (
        "best_val_loss",
        "test loss",
        "test mAP",
        "test precision",
        "test recall",
    )
    sort_key = "training set size (numerical)"

    df = (
        q.select(
            [
                *cols,
                pl.col("training set size")
                .str.replace(" images", "")
                .cast(pl.Int32)
                .alias(sort_key),
            ]
        )
        .collect()
        .groupby("Notes")
        .agg(pl.col(c).mean() for c in (*numerical_cols, sort_key))
        .sort(sort_key)
    )
    print(df)

    plotting_columns = (
        ["test loss", "best_val_loss"],
        ["test mAP", "test precision", "test recall"],
    )

    fig, ax = plt.subplots(len(plotting_columns), 1, figsize=(10, 10))
    fig.suptitle("Model Performance vs. Training Set Size")

    for i in range(len(plotting_columns)):
        for col in plotting_columns[i]:
            ax[i].set_ylabel(col)
            ax[i].plot(
                df.select(sort_key),
                df.select(col),
                label=col,
            )
        ax[i].legend()

    plt.legend()
    fig.tight_layout()
    plt.show()
