#! /usr/bin/env python3

import sys
import zarr
import math

from PIL import Image
from typing import List
from pathlib import Path
from functools import partial

from lfm_data_utilities.utils import multithread_map_unordered, get_list_of_zarr_files


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


def check_num_imgs_is_num_zarr_imgs(path_to_zarr_zip: Path) -> Path:
    data = zarr.open(str(path_to_zarr_zip), "r")
    data_len = data.initialized if isinstance(data, zarr.Array) else len(data)

    image_dir = path_to_zarr_zip.parent / "images"
    num_imgs = len(list(image_dir.iterdir())) if image_dir.exists() else 0

    if num_imgs != data_len:
        print(
            f"num images in {image_dir} ({num_imgs}) != num images in zarr file ({data_len})"
        )
        return path_to_zarr_zip


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "convert a set of run folders zip files to image folders"
    )
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "check if the number of images in the images folder "
            "matches the number of images in the zarr file",
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "fix the number of images in the images folder to match "
            "the number of images in the zarr file (equiv to --check "
            "and then --overwrite for mismatched folders only)",
        ),
    )
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

    files = get_list_of_zarr_files(run_set)

    if len(files) == 0:
        raise ValueError(f"no zarr files found in directory {sys.argv[1]}")

    if args.fix:
        files_to_fix = list(filter(
            bool,
            multithread_map_unordered(
                files, check_num_imgs_is_num_zarr_imgs, verbose=False
            ),
        ))
        multithread_map_unordered(
            files_to_fix,
            partial(convert_zarr_to_image_folder, skip=False),
            max_num_threads=4,
        )
    elif args.check:
        multiprocess_fn(
            files, check_num_imgs_is_num_zarr_imgs, ordered=False, verbose=False
        )
    else:
        multiprocess_fn(
            files, partial(convert_zarr_to_image_folder, skip=skip), ordered=False
        )
