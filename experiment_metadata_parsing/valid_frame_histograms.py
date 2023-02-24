import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_list_of_per_image_metadata_files,
    get_list_of_experiment_level_metadata_files,
    parse_csv,
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
    print(f"{len(metadata_files)} per image metadata files found")

    histogram_keys = {"focus", "flowrate"}
    valid_frame_counts = dict.fromkeys(histogram_keys, [])

    for file in metadata_files:
        data = parse_csv(file)
        if  bool(data) and int(data["im_counter"][-1]) >= IMCOUNT_TARGET:
            valid_frame_counts["focus"].append(count_valid_focus_frames(data["focus_error"]))
            valid_frame_counts["flowrate"].append(count_valid_focus_frames(data["flowrate"]))

    print(valid_frame_counts)

    valid_focus_histogram, focus_bin_edges = np.histogram(valid_frame_counts["focus"], bins=20)
    focus_bin_centers = [(a + b) / 2 for a, b in zip(focus_bin_edges[0:-1], focus_bin_edges[1:])]

    valid_flowrate_histogram, flowrate_bin_edges = np.histogram(valid_frame_counts["focus"], bins=20)
    flowrate_bin_centers = [(a + b) / 2 for a, b in zip(flowrate_bin_edges[0:-1], flowrate_bin_edges[1:])]

    # Creating histogram
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize =(10, 7))

    axs[0].bar(focus_bin_centers, valid_focus_histogram, width=100)
    axs[1].bar(flowrate_bin_centers, valid_flowrate_histogram, width=100)

    fig.suptitle(title)
    axs[0].set_title(f"Focus within range {MIN_FOCUS_TARGET, MAX_FOCUS_TARGET} steps")
    axs[1].set_title(f"Flowrate within range {MIN_FLOWRATE_TARGET, MAX_FLOWRATE_TARGET} FoVs / sec")

    # Show plot
    plt.show()

def count_valid_focus_frames(data):
    # empty = 0
    # good = 0
    # bad = 0
    # for val in data:
    #     if bool(val):
    #         empty += 1
    #     elif MIN_FOCUS_TARGET < int(val) < MAX_FOCUS_TARGET:
    #         good += 1
    #     else:
    #         bad += 1

    # print(f"EMPTY {empty} GOOD {good} BAD {bad}")

    return sum(1 for val in data if val != "" and MIN_FOCUS_TARGET < int(val) < MAX_FOCUS_TARGET)

def count_valid_flowrate_frames(data):
    return sum(1 for val in data if val != "" and MIN_FLOWRATE_TARGET < float(val) < MAX_FLOWRATE_TARGET)

if __name__ == "__main__":
    try:
        path =sys.argv[1]
    except IndexError as e:
        raise Exception("Expected format 'python valid_frame_histograms <filepath> [title]'")

    try: 
        title = sys.argv[2]
    except IndexError:
        title = path

    plot_valid_frame_histograms(path, title)