import argparse
import numpy as np

from typing import List, Optional, Union, Tuple
from pathlib import Path

from lfm_data_utilities.utils import (
    get_list_of_per_image_metadata_files,
    get_corresponding_ssaf_file,
    multiprocess_load_csv,
)
from histogram_constants import (
    IMCOUNT_TARGET,
)


def run(metadata_dir, ssaf_dir, title, output=None) -> None:
    metadata_files, ssaf_files = get_all_files(metadata_dir, ssaf_dir)

    valid_focus_percs = []
    valid_flowrate_percs = []

    # Get % good frames for each dataset
    data_files = multiprocess_load_csv(metadata_files)
    for data in data_files:
        vals = data["vals"]
        if bool(vals) and int(vals["im_counter"][-1]) >= IMCOUNT_TARGET:
            valid_focus_percs.append(count_valid_focus_frames(vals["focus_error"]))
            valid_flowrate_percs.append(
                count_valid_frames(vals["flowrate"], data["filepath"])
            )


#    print(valid_focus_percs)
#    print(valid_flowrate_percs)

# # Filter out nan
# filtered_valid_focus_percs = valid_focus_percs[np.isnan()]
# filtered_valid_flowrate_percs = filter_nonetype(valid_flowrate_percs)

# valid_focus_histogram, focus_bin_edges = np.histogram(
#     filtered_valid_focus_percs, bins=20
# )
# focus_bin_centers = [
#     (a + b) / 2 for a, b in zip(focus_bin_edges, focus_bin_edges[1:])
# ]

# valid_flowrate_histogram, flowrate_bin_edges = np.histogram(
#     filtered_valid_flowrate_percs, bins=20
# )
# flowrate_bin_centers = [
#     (a + b) / 2 for a, b in zip(flowrate_bin_edges[0:-1], flowrate_bin_edges[1:])
# ]

# fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 7))

# axs[0].bar(focus_bin_centers, valid_focus_histogram)
# axs[1].bar(flowrate_bin_centers, valid_flowrate_histogram)

# fig.suptitle(title)
# axs[0].set_title(f"Focus within range {MIN_FOCUS_TARGET, MAX_FOCUS_TARGET} steps")
# axs[0].set_xlabel("% valid frames out of all focus measurements")
# axs[0].set_ylabel("Number of datasets")

# axs[1].set_title(
#     f"Flowrate within range {MIN_FLOWRATE_TARGET, MAX_FLOWRATE_TARGET} FoVs / sec"
# )
# axs[1].set_xlabel("% valid frames out of all flowrate measurements")
# axs[1].set_ylabel("Number of datasets")

# plt.show()


def get_all_files(metadata_dir: str, ssaf_dir: str) -> Tuple[List[Path], List[Path]]:
    metadata_files = get_list_of_per_image_metadata_files(metadata_dir)

    ssaf_files = filter_nonetype(
        [
            get_corresponding_ssaf_file(metadata_file, ssaf_dir)
            for metadata_file in metadata_files
        ]
    )

    print(f"{len(metadata_files)} per image metadata files found", flush=True)
    print(f"{len(ssaf_files)} SSAF files found", flush=True)

    return metadata_files, ssaf_files


def count_valid_focus_frames(
    data: List[Optional[Union[float, int]]],
    min_target: Optional[Union[float, int]],
    max_target: Optional[Union[float, int]],
) -> List[Optional[Union[float, int]]]:
    ready = True

    good = 0
    total = 0

    for focus_val in data:
        if focus_val:
            if ready:
                ready = False
                total += 1
                if min_target < float(focus_val) < max_target:
                    good += 1
        else:
            ready = True

    return good / total * 100


def count_valid_frames(
    data: List[Optional[Union[float, int]]],
    min_target: Optional[Union[float, int]],
    max_target: Optional[Union[float, int]],
    file: str,
) -> List[Optional[Union[float, int]]]:
    good = sum(1 for val in data if val != "" and min_target < float(val) < max_target)
    total = sum(1 for val in data if val != "")

    if total == 0:
        print(f"No measurements for {file}")
        return np.nan

    return good / total * 100


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--metadata",
        help="Directory containing metadata .csv files",
        required=True,
    )
    argparser.add_argument(
        "-s", "--ssaf", help="Directory containing SSAF .txt files", required=True
    )
    argparser.add_argument("-t", "--title", help="Title for plot")
    argparser.add_argument("-o", "--output", help="Filename to export plot to")

    args = argparser.parse_args()

    if args.title:
        title = args.title
    else:
        title = args.metadata

    if args.output:
        run(args.metadata, args.ssaf, title, output=args.output)
    else:
        run(args.metadata, args.ssaf, title)
