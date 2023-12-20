#! /usr/bin/env python3

import sys
import zarr
import math

from PIL import Image
from pathlib import Path
from typing import Optional
from functools import partial

from lfm_data_utilities.utils import (
    multiprocess_fn,
    multithread_map_unordered,
    get_list_of_zarr_files,
    path_relative_to,
)


def convert_zarr_to_image_folder(
    path_to_zarr_zip: Path,
    skip: bool = True,
    image_runset_dir: Optional[Path] = None,
    path_to_runset: Optional[Path] = None,
):
    """
    this will convert the specified zarr file to a folder of images. if the
    folder of images already exists and `skip` is true, this skips that
    conversion.

    If `image_runset_dir` is none, this will create a folder 'images' beside the
    zarr folder. Otherwise, image_runset_dir will hold the images. The folder
    structure of `path_to_runset` will be copied. `image_runset_dir` and `path_to_runset`
    must both have a value or must both be None
    """
    paths_are_valid = (image_runset_dir is None) == (path_to_runset is None)
    if not paths_are_valid:
        raise ValueError(
            "image_runset_dir and path_to_runset must both have a value or must both be None"
        )

    data = zarr.open(str(path_to_zarr_zip), mode="r")

    if image_runset_dir is None:
        image_dir = path_to_zarr_zip.parent / "images"
    else:
        assert path_to_runset is not None  # for mypy
        image_dir = (
            image_runset_dir / path_relative_to(path_to_zarr_zip, path_to_runset)
        ).parent / "images"

    if image_dir.exists() and len(list(image_dir.iterdir())) > 0 and skip:
        print(f"skipping {image_dir} because images already exist!")
        return
    elif image_dir.exists() and not skip:
        # being explicit w/ condition above because it would be bad to make a mistake here
        for img_path in image_dir.glob("*.png"):
            img_path.unlink()  # remove file

    # we converted storing data as a zarr.Group to a zarr.Array, so we need
    # to manage both cases
    data_len = data.initialized if isinstance(data, zarr.Array) else len(data)
    if data_len == 0:
        return

    image_dir.mkdir(parents=True, exist_ok=True)

    N = int(math.log(data_len, 10) + 1)

    def cp_img_i(i):
        img = data[:, :, i] if isinstance(data, zarr.Array) else data[i][:]
        Image.fromarray(img).save(image_dir / f"img_{i:0{N}}.png")

    multithread_map_unordered(range(data_len), cp_img_i, verbose=False)


def check_num_imgs_is_num_zarr_imgs(
    path_to_zarr_zip: Path,
    image_runset_dir: Optional[Path] = None,
    path_to_runset: Optional[Path] = None,
) -> Optional[Path]:
    data = zarr.open(str(path_to_zarr_zip), "r")
    data_len = data.initialized if isinstance(data, zarr.Array) else len(data)

    paths_are_valid = (image_runset_dir is None) == (path_to_runset is None)
    if not paths_are_valid:
        raise ValueError(
            "image_runset_dir and path_to_runset must both have a value or must both be None"
        )

    if image_runset_dir is None:
        image_dir = path_to_zarr_zip.parent / "images"
    else:
        assert path_to_runset is not None  # for mypy
        image_dir = (
            image_runset_dir / path_relative_to(path_to_zarr_zip, path_to_runset)
        ).parent / "images"

    num_imgs = len(list(image_dir.iterdir())) if image_dir.exists() else 0

    if num_imgs != data_len:
        print(
            f"num images in {image_dir} ({num_imgs}) != num images in zarr file ({data_len})"
        )
        return path_to_zarr_zip
    # mypy really wants explicit 'return None'
    return None


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
            "matches the number of images in the zarr file"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help=(
            "optional image directory for the folders of images - the folder structure for "
            "path_to_runsets will be mimicked (defaults to the same directory as the zarr files)"
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "fix the number of images in the images folder to match "
            "the number of images in the zarr file (equiv to --check "
            "and then --overwrite for mismatched folders only)"
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

    check_fcn = partial(
        check_num_imgs_is_num_zarr_imgs,
        image_runset_dir=args.image_dir,
        path_to_runset=args.path_to_runset if args.image_dir is not None else None,
    )

    if args.fix:
        files_to_fix = list(
            filter(
                bool,
                multithread_map_unordered(files, check_fcn, verbose=False),
            )
        )
        multiprocess_fn(
            files_to_fix,
            partial(
                convert_zarr_to_image_folder,
                image_runset_dir=args.image_dir,
                path_to_runset=(
                    args.path_to_runset if args.image_dir is not None else None
                ),
                skip=False,
            ),
            ordered=False,
        )
    elif args.check:
        multithread_map_unordered(files, check_fcn, verbose=False)
    else:
        multiprocess_fn(
            files,
            partial(
                convert_zarr_to_image_folder,
                image_runset_dir=args.image_dir,
                path_to_runset=(
                    args.path_to_runset if args.image_dir is not None else None
                ),
                skip=skip,
            ),
            ordered=False,
        )
