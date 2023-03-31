#! /usr/bin/env python3

import sys
import zarr
import math

from PIL import Image
from pathlib import Path
from functools import partial

from utils import multiprocess_directory_work


def convert_zarr_to_image_folder(path_to_zarr_zip: Path, skip=True):
    data = zarr.open(str(path_to_zarr_zip))

    image_dir = path_to_zarr_zip.parent / "images"

    if image_dir.exists() and skip:
        print(f"skipping {image_dir} because images already exist!")
        return

    data_len = data.initialized if hasattr(data, "nchunks") else len(data)
    if data_len == 0:
        return

    image_dir.mkdir(parents=True, exist_ok=True)

    N = int(math.log(data_len, 10) + 1)

    for i in range(data_len):
        img = data[:, :, i] if hasattr(data, "nchunks") else data[i][:]
        Image.fromarray(img).save(image_dir / f"img_{i:0{N}}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "convert a set of run folders zip files to image folders"
    )
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument(
        "--existing-label-action",
        "-e",
        choices=["skip", "overwrite"],
        default="skip",
        help="skip or overwrite existing labels when encountered, defaults to 'skip'",
    )
    args = parser.parse_args()

    run_set = args.path_to_runset
    skip = args.existing_label_action == "skip"

    if not run_set.exists():
        raise FileNotFoundError(f"directory {sys.argv[1]} not found")

    files = list(run_set.glob("./**/*.zip"))

    if len(files) == 0:
        raise ValueError(f"no zarr files found in directory {sys.argv[1]}")

    multiprocess_directory_work(files, partial(convert_zarr_to_image_folder, skip=skip))
