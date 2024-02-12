import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def filter_for_cells(arr):
    cells = [0, 1, 2, 3]
    return arr[:, np.isin(arr[6, :], cells)]


def filter_for_conf(arr, conf_thresh):
    return arr[:, arr[7, :] > conf_thresh]


def get_healthy(arr):
    filtered = filter_for_conf(filter_for_cells(arr), 0.9)
    healthy = filtered[:, filtered[6, :] == 0]

    return healthy


def get_dims(arr):
    widths = arr[3, :] - arr[1, :]
    heights = arr[4, :] - arr[2, :]
    areas = widths * heights

    return widths, heights, areas


def plot_heights_and_widths(arr, filename, output_dir):
    title = f"{filename} - healthy thumbnail sizes"

    widths, heights, areas = get_dims(arr)

    fig = plt.figure(figsize=(10, 7), dpi=125)
    fig.suptitle(f"{title}")
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.hist(widths, edgecolor="k", bins=40)
    ax1.set_title("Histogram of thumbnail widths")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, 90)

    ax2.hist(heights, edgecolor="k", color="C6", bins=40)
    ax2.set_title("Histogram of thumbnail heights")
    ax2.set_xlabel("Width (px)")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 90)

    ax3.hist(widths * heights, edgecolor="k", color="C7", bins=40)
    ax3.set_title("Histogram of thumbnail areas")
    ax3.set_xlabel("Width (px)")
    ax3.set_ylabel("Count")
    ax3.set_xlim(0, 4500)

    plt.tight_layout()

    plt.savefig(f"{output_dir}/{filename}_healthy_sizes.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot size of healthy cells from the given npy file."
    )
    parser.add_argument("npy_file", type=str, help="Path to the npy file.")
    parser.add_argument(
        "output_dir", type=str, help="Path to save matplotlib figure (.jpg)"
    )

    args = parser.parse_args()

    # Load the npy file as a numpy array
    filename = Path(args.npy_file).stem
    npy_data = np.load(args.npy_file)

    output_dir = args.output_dir

    healthy = get_healthy(npy_data)
    widths, heights, areas = get_dims(healthy)

    mean = np.mean(areas)
    sd = np.std(areas)

    plot_heights_and_widths(healthy, filename, output_dir)
