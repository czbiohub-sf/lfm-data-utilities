import argparse
import multiprocessing as mp
import sys

from tqdm import tqdm
from pathlib import Path

from lfm_data_utilities import utils


def open_and_parse_ssaf_txt(filepath: Path):
    with open(filepath, "r") as f:
        vals = f.readlines()
        vals = [float(x.strip()) for x in vals]
    return vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Make video from a zarr file",
    )
    parser.add_argument(
        "path_to_folder",
        type=Path,
        help="Path to the folder containing zarr files (will recursively search the given directory for zarr files)",
    )
    parser.add_argument("save_path", type=Path, help="Path to save videos")
    parser.add_argument(
        "--img_counter",
        type=bool,
        help="Add img counter text (default True)",
        default=True,
    )
    parser.add_argument(
        "--ssaf1_vals",
        type=Path,
        help="Path to SSAF values (first model)",
        default=None,
    )
    parser.add_argument(
        "--ssaf1_name", type=Path, help="Name of SSAF model 1", default=""
    )
    parser.add_argument(
        "--ssaf2_vals",
        type=Path,
        help="Path to SSAF values (can pass in a second model's inferences too)",
        default=None,
    )
    parser.add_argument(
        "--ssaf2_name", type=Path, help="Name of SSAF model 2", default=""
    )

    args = parser.parse_args()

    path_to_runset = args.path_to_folder
    path_to_save = Path(args.save_path)

    ssaf1_vals = (
        open_and_parse_ssaf_txt(args.ssaf1_vals)
        if args.ssaf1_vals is not None
        else None
    )
    ssaf1_name = args.ssaf1_name

    ssaf2_vals = (
        open_and_parse_ssaf_txt(args.ssaf2_vals)
        if args.ssaf2_vals is not None
        else None
    )
    ssaf2_name = args.ssaf2_name

    datasets = utils.load_datasets(path_to_runset, fail_silently=True)
    valid_datasets = [d for d in datasets if d.successfully_loaded]

    print("Generating videos...")
    with mp.Pool() as pool:
        pool.starmap(
            utils.make_video,
            [
                (
                    x,
                    path_to_save,
                    1,
                    ssaf1_name,
                    ssaf1_vals,
                    ssaf2_name,
                    ssaf2_vals,
                    args.img_counter,
                )
                for x in tqdm(valid_datasets)
            ],
        )
