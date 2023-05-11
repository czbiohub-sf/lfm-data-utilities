import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfm_data_utilities.utils import parse_datetime_string, load_txtfile


def run(file_dir, txtfile_dir=None):
    # Get date
    basename = pathlib.Path(file_dir).stem
    print(basename)
    date = parse_datetime_string(basename)

    # Read data
    data = pd.read_csv(file_dir)
    raw_w_nan = data["focus_error"].to_numpy(na_value=np.nan)
    filtered_w_nan = data["filtered_focus_error"].to_numpy(na_value=np.nan)

    # Get data without nans
    non_nan = ~np.isnan(raw_w_nan)
    raw = raw_w_nan[non_nan]
    filtered = filtered_w_nan[non_nan]

    print(np.argwhere(~np.isnan(raw_w_nan)))

    # Get throttle
    throttle = np.diff(np.where(non_nan is True))[0][0]

    # Get adjustments
    adjusted = data["focus_adjustment"].to_numpy(na_value=0).astype(bool)[non_nan]

    # Plot
    plt.plot(raw, label="Raw error", alpha=0.5, color="orange")
    plt.plot(filtered, label="Filtered error", alpha=0.5, color="green")
    plt.scatter(
        np.where(adjusted == 1),
        filtered[adjusted],
        label="Focus adjustment",
        color="green",
    )

    # Labels
    plt.xlabel(f"Measurement (every {throttle} frames)")
    plt.ylabel("SSAF error [motor steps]")
    plt.title(f"{date}: SSAF measured every {throttle} frames")

    # Display
    plt.legend()
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-f", "--file", help="Path to per image metadata file", required=True
    )
    argparser.add_argument(
        "-t", "--txt", help="Path to txtfile with SSAF data for every frame",
    )

    args = argparser.parse_args()
    run(args.file, txtfile_dir=args.txt)
