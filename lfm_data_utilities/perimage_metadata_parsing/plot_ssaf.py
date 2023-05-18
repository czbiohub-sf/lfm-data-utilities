import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfm_data_utilities.utils import get_rms


def run(file_dir, txtfile_dir=None):
    # Get folder name
    basename = pathlib.Path(file_dir).parent.stem

    # Read data
    data = pd.read_csv(file_dir)

    # Extract each dataset
    raw_w_nan = data["focus_error"].to_numpy(na_value=np.nan)
    try:
        filtered_w_nan = data["filtered_focus_error"].to_numpy(na_value=np.nan)
        adjusted = data["focus_adjustment"].to_numpy(na_value=0).astype(bool)

        filtered_available = True
        err_label = "Raw error"
    except KeyError:
        filtered_available = False
        err_label = "Batched error"
    if txtfile_dir is not None:
        txt_data = load_txtfile(txtfile_dir)

    # Get nan filters
    non_nan = ~np.isnan(raw_w_nan)

    # Get data without nan
    raw = raw_w_nan[non_nan]
    if filtered_available:
        filtered = filtered_w_nan[non_nan]

    # Get frame index
    frame_index = np.transpose(np.nonzero(non_nan))
    if filtered_available:
        adjusted_index = np.transpose(np.nonzero(adjusted)) 

    # Get throttle
    throttle = frame_index[1][0] - frame_index[0][0]

    # Plot
    if txtfile_dir is not None:
        plt.plot(txt_data,  label="Post-processed", alpha=0.5, color="gold")

    plt.plot(frame_index, raw, label="Raw error", alpha=0.5, color="brown")

    if filtered_available:
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
    if txtfile_dir is not None:
        plt.title(f"{basename}: SSAF measured every {throttle} frames (RMS = {get_rms(txt_data)})")
    else:
        plt.title(f"{basename}: SSAF measured every {throttle} frames (RMS = {get_rms(raw)})")

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
