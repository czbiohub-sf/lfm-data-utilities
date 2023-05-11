import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(file_dir, txtfile_dir=None):
    # Get folder name
    basename = pathlib.Path(file_dir).parent.stem

    # Read data
    data = pd.read_csv(file_dir)

    # Extract each dataset
    raw_w_nan = data["focus_error"].to_numpy(na_value=np.nan)
    filtered_w_nan = data["filtered_focus_error"].to_numpy(na_value=np.nan)
    adjusted = data["focus_adjustment"].to_numpy(na_value=0).astype(bool)

    # Get nan filters
    non_nan = ~np.isnan(raw_w_nan)

    # Get data without nan
    raw = raw_w_nan[non_nan]
    filtered = filtered_w_nan[non_nan]

    # Get frame index
    frame_index = np.transpose(np.nonzero(non_nan))
    adjusted_index = np.transpose(np.nonzero(adjusted))

    # Get throttle
    throttle = frame_index[1][0] - frame_index[0][0]

    # Plot
    plt.plot(frame_index, raw, label="Raw error", alpha=0.5, color="orange")
    plt.plot(frame_index, filtered, label="Filtered error", alpha=0.5, color="green")
    plt.scatter(
        adjusted_index,
        filtered_w_nan[adjusted],
        label="Focus adjustment",
        color="green",
    )

    # Labels
    plt.xlabel(f"Measurement (every {throttle} frames)")
    plt.ylabel("SSAF error [motor steps]")
    plt.title(f"{basename}: SSAF measured every {throttle} frames")

    # Display
    plt.legend()
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-f", "--file", help="Path to per image metadata file", required=True
    )
    argparser.add_argument(
        "-t",
        "--txt",
        help="Path to txtfile with SSAF data for every frame",
    )

    args = argparser.parse_args()
    run(args.file, txtfile_dir=args.txt)
