#! /usr/bin/env python3

import sys
import zarr
import math

from PIL import Image
from pathlib import Path
from functools import partial

from utils import multiprocess_directory_work


def convert_zarr_to_image_folder(path_to_zarr_zip: Path, skip=True):
    data = zarr.open(str(path_to_zarr_zip), "r")

    image_dir = path_to_zarr_zip.parent / "images"

    if image_dir.exists() and len(list(image_dir.iterdir())) > 0 and skip:
        print(f"skipping {image_dir} because images already exist!")
        return
    elif image_dir.exists() and not skip:
        # being explicit w/ condition above because it would be bad to make a mistake here
        for img_path in image_dir.glob("*.png"):
            img_path.unlink()  # remove file

    # we converted storing data as a zarr.Group to a zarr.Array
    data_len = data.initialized if isinstance(data, zarr.Array) else len(data)
    if data_len == 0:
        return

    image_dir.mkdir(parents=True, exist_ok=True)

    N = int(math.log(data_len, 10) + 1)

    for i in range(data_len):
        img = data[:, :, i] if isinstance(data, zarr.Array) else data[i][:]
        Image.fromarray(img).save(image_dir / f"img_{i:0{N}}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "convert a set of run folders zip files to image folders"
    )
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument(
        "--existing-image-action",
        "-e",
        choices=["skip", "overwrite"],
        default="skip",
        help="skip or overwrite existing images when encountered, defaults to 'skip'",
    )
    args = parser.parse_args()

    run_set = args.path_to_runset
    skip = args.existing_image_action == "skip"

    if not run_set.exists():
        raise FileNotFoundError(f"directory {sys.argv[1]} not found")

    files = [f for f in run_set.glob("./**/*.zip") if not Path(f).name.startswith(".")]

    if len(files) == 0:
        raise ValueError(f"no zarr files found in directory {sys.argv[1]}")

    multiprocess_directory_work(files, partial(convert_zarr_to_image_folder, skip=skip))
