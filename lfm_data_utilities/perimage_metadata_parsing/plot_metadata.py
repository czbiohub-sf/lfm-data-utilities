"""
Plot parameters of "per_image" metadata files.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import List, Dict

from lfm_data_utilities import utils


# for reference
# keys(['im_counter', 'timestamp', 'motor_pos', 'pressure_hpa', 'pressure_status_flag',
# # 'syringe_pos', 'flowrate', 'focus_error', 'temperature', 'humidity', 'looptime',
# # 'runtime', 'zarrwriter_qsize']


def plot(data, scope_name: str):
    """Plot flowrate / focus error / syringe position / motor position / total number of images / cell counts"""

    fig = plt.figure(12, figsize=(10, 7))
    gs = gridspec.GridSpec(3, 4)
    fig.suptitle(f"{scope_name}, num_datasets={len(data)}")

    flowrates: List[List[float]] = []
    focus_errs: List[List[float]] = []
    syringe_pos: List[List[float]] = []
    motor_pos: List[List[float]] = []
    cumulative_cell_counts: List[List[float]] = []

    param_data = [flowrates, focus_errs, syringe_pos, motor_pos, cumulative_cell_counts]

    for i, param in enumerate(
        ["flowrate", "focus_error", "syringe_pos", "motor_pos", "cell_count_cumulative"]
    ):
        for d in data:
            if param in d:  # Since cell_count_cumulative may not be in old datasets
                param_data[i].append([float(x) for x in d[param] if x != ""])

    ax_fr = plt.subplot(gs[0, :2])
    [ax_fr.plot(flowrate, alpha=0.5) for flowrate in flowrates]
    ax_fr.set_title("Flowrate")

    ax_focus = plt.subplot(gs[0, 2:])
    [ax_focus.plot(np.diff(focus_err), alpha=0.5) for focus_err in focus_errs]
    ax_focus.set_title("Focus error")

    ax_syringe = plt.subplot(gs[1, :2])
    [ax_syringe.plot(sp, alpha=0.5) for sp in syringe_pos]
    ax_syringe.set_title("Syringe position")

    ax_motor = plt.subplot(gs[1, 2:])
    [ax_motor.plot(mp, alpha=0.5) for mp in motor_pos]
    ax_motor.set_title("Motor position")

    ax_run_length = plt.subplot(gs[2, 0:2])
    ax_run_length.bar(range(len(data)), [len(d["im_counter"]) for d in data])
    ax_run_length.set_title("Run length (# of images)")
    ax_run_length.set_xlabel("Dataset idx")

    cc = [(i, c) for i, c in enumerate(cumulative_cell_counts) if len(c) > 0]
    ax_cell_count = plt.subplot(gs[2, 2:4])
    ax_cell_count.bar([x[0] for x in cc], [x[1][-1] for x in cc])
    ax_cell_count.set_title("Cumulative cell counts")
    ax_cell_count.set_ylabel("Number of cells")

    plt.tight_layout()
    plt.show(block=False)
    input("Press enter to advance: ")
    plt.close()


def plot_many_sorted_by_day():
    """Plot in aggregate sorted by day (i.e all runs from the same day will be on the same plot).

    Allow user to select an optional date range.
    """

    # Get metadata files
    tld = input("Enter top level directory: ")
    top_level_folders = utils.get_list_of_oracle_run_folders(tld)
    csv_filepaths = utils.get_list_of_per_image_metadata_files(tld)

    # Get and display dates
    print("Data was collected on the following days: ")
    dates = utils.get_dates_from_top_level_folders(top_level_folders)
    [print(d) for d in dates]

    d1, d2 = utils.get_date_range_from_user()
    dates = [d for d in dates if d1 <= utils.parse_datetime_string(d, "%Y-%m-%d") <= d2]

    scope_name = input("Enter the scope name: ")
    for date in dates:
        csvs_from_same_day = utils.get_all_metadata_files_from_same_day(
            csv_filepaths, utils.parse_datetime_string(date, "%Y-%m-%d")
        )

        print(f"Loading {len(csvs_from_same_day)} datasets from {date}...")
        data: List[Dict] = []
        data = utils.multiprocess_load_csv(csvs_from_same_day)

        data = [d["vals"] for d in data if len(d["vals"].keys()) > 0]
        print(f"Plotting {len(data)} valid datasets...")

        plot_title = scope_name + "_" + date
        plot(data, plot_title)


def plot_single_folder():
    """Plots all the experiments in a single folder."""
    # Get metadata files
    tld = input("Enter folder path: ")
    csv_filepaths = utils.get_list_of_per_image_metadata_files(tld)
    scope_name = input("Enter the scope name: ")

    print(f"Loading {len(csv_filepaths)} datasets...")
    data = utils.multiprocess_load_csv(csv_filepaths)
    data = [d["vals"] for d in data if len(d["vals"].keys()) > 0]

    print(f"Plotting {len(data)} valid datasets...")
    plot(data, scope_name)


def main():
    print("===PLOTTING TOOL FOR EXPERIMENT METADATA===")
    print(f"{'='*10}")
    print("INSTRUCTIONS")
    print(
        "- Pass in a top-level directory containing multiple oracle runs (each run of which may contain multiple experiments)."
    )
    print(
        "- Or pass in a single oracle run folder, or more specific still, a single experiment."
    )
    print(
        "- If passing in a top-level directory containing multiple days worth of data, you will be prompted to enter an optional date range."
    )
    print(
        "- Plots will then be generated. All the runs on a particular day will be aggregated together."
    )
    print(f"{'='*10}")
    user_input = input(
        "Enter 1 for aggregated by day, 2 for single folder/experiment: "
    )
    if user_input == "1":
        plot_many_sorted_by_day()
    elif user_input == "2":
        plot_single_folder()


if __name__ == "__main__":
    main()
