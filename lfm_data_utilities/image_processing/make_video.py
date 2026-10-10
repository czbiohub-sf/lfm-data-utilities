import argparse
from pathlib import Path

from lfm_data_utilities import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Make video from a zarr file",
    )
    parser.add_argument(
        "path_to_folder",
        type=Path,
        help="Path to the folder containing zarr files (will recursively search the given directory for zarr files). Videos are saved next to each zarr file.",
    )
    parser.add_argument(
        "--img-counter",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Add img counter text (default True)",
    )
    parser.add_argument(
        "--contrast-boost",
        action="store_true",
        help="Stretch pixel range [190, 255] to [0, 255]; saved as <name>-contrast-boost.avi",
    )

    args = parser.parse_args()

    datasets = utils.load_datasets(args.path_to_folder, fail_silently=True)
    valid_datasets = [d for d in datasets if d.successfully_loaded]

    print(f"Generating {len(valid_datasets)} videos...")
    for i, d in enumerate(valid_datasets, 1):
        print(f"({i}/{len(valid_datasets)}) ", end="", flush=True)
        utils.make_video(
            d, add_im_counter_text=args.img_counter, contrast_boost=args.contrast_boost
        )
