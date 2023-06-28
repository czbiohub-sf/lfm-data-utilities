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
        "Name",
        "Notes",
        "best_val_loss",
        "classification_loss",
        "iou_loss",
        "objectnes_loss_no_obj",
        "objectnes_loss_obj",
        "test loss",
        "test mAP",
        "test precision",
        "test recall",
        "val loss",
    )

    dataset_sizes = (
        q.select("training set size")
        .collect()
        .apply(lambda g: int(g[0].replace(" images", "")))
        .rename({"apply": "training set size"})  # probably hacky
    )
    rest = q.select(cols).collect()
    df = dataset_sizes.hstack(rest)

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Model Performance vs. Training Set Size")

    plotting_columns = (
        ["test loss", "val loss"],
        ["test mAP", "test precision", "test recall"],
    )

    for i in range(2):
        ax[i].plot(
            df.select("training set size"),
            df.select(plotting_columns[i]),
            label=plotting_columns[i],
        )
        ax[i].legend()

    fig.tight_layout()
    plt.show()
