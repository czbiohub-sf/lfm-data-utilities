import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_list_of_per_image_metadata_files,
    multiprocess_load_csv,
)
from constants import (
    IMCOUNT_TARGET,
    MIN_FOCUS_TARGET,
    MAX_FOCUS_TARGET,
    MIN_FLOWRATE_TARGET,
    MAX_FLOWRATE_TARGET,
)


def plot_valid_frame_histograms(path, title):
    metadata_files = get_list_of_per_image_metadata_files(path)
    print(f"{len(metadata_files)} per image metadata files found", flush=True)

    valid_focus_percs = []
    valid_flowrate_percs = []

    # Get % good frames for each dataset
    data_files = multiprocess_load_csv(metadata_files)
    for data in data_files:
        vals = data["vals"]
        if bool(vals) and int(vals["im_counter"][-1]) >= IMCOUNT_TARGET:
            valid_focus_perc = count_valid_focus_frames(vals["focus_error"])
            valid_focus_percs.append(valid_focus_perc)

            valid_flowrate_perc = count_valid_flowrate_frames(
                vals["flowrate"], data["filepath"]
            )
            valid_flowrate_percs.append(valid_flowrate_perc)

    # Filter out nan
    filtered_valid_focus_percs = [val for val in valid_focus_percs if not np.isnan(val)]
    filtered_valid_flowrate_percs = [
        val for val in valid_flowrate_percs if not np.isnan(val)
    ]

    valid_focus_histogram, focus_bin_edges = np.histogram(
        filtered_valid_focus_percs, bins=20
    )
    focus_bin_centers = [
        (a + b) / 2 for a, b in zip(focus_bin_edges[0:-1], focus_bin_edges[1:])
    ]

    valid_flowrate_histogram, flowrate_bin_edges = np.histogram(
        filtered_valid_flowrate_percs, bins=20
    )
    flowrate_bin_centers = [
        (a + b) / 2 for a, b in zip(flowrate_bin_edges[0:-1], flowrate_bin_edges[1:])
    ]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 7))

    axs[0].bar(focus_bin_centers, valid_focus_histogram)
    axs[1].bar(flowrate_bin_centers, valid_flowrate_histogram)

    fig.suptitle(title)
    axs[0].set_title(f"Focus within range {MIN_FOCUS_TARGET, MAX_FOCUS_TARGET} steps")
    axs[0].set_xlabel("% valid frames out of all focus measurements")
    axs[0].set_ylabel("Number of datasets")

    axs[1].set_title(
        f"Flowrate within range {MIN_FLOWRATE_TARGET, MAX_FLOWRATE_TARGET} FoVs / sec"
    )
    axs[1].set_xlabel("% valid frames out of all flowrate measurements")
    axs[1].set_ylabel("Number of datasets")

    plt.show()


def count_valid_focus_frames(focus_data):
    ready = True

    good = 0
    total = 0

    for focus_val in focus_data:
        if focus_val:
            if ready:
                ready = False
                total += 1
                if MIN_FOCUS_TARGET < float(focus_val) < MAX_FOCUS_TARGET:
                    good += 1
        else:
            ready = True

    return good / total * 100


def count_valid_flowrate_frames(data, file):
    good = sum(
        1
        for val in data
        if val != "" and MIN_FLOWRATE_TARGET < float(val) < MAX_FLOWRATE_TARGET
    )
    total = sum(1 for val in data if val != "")

    if total == 0:
        print(f"No flowrate measurements for {file}")
        return np.nan

    return good / total * 100


if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except IndexError as e:
        raise Exception(
            "Expected format 'python valid_frame_histograms <filepath> [title]'"
        )

    try:
        title = sys.argv[2]
    except IndexError:
        title = path

    plot_valid_frame_histograms(path, title)
