#! /usr/bin/env python3

import re
import shutil

from pathlib import Path
from functools import partial

from lfm_data_utilities.utils import (
    multiprocess_fn,
    multithread_map_unordered,
    is_not_hidden_path,
)


def does_the_path_match(pattern: str, path: Path) -> bool:
    return re.fullmatch(pattern, path.name) is not None


def move_images(
    path_to_image_dir: Path,
    path_to_image_subsets_dir: Path,
    regex_str: str,
    min_num_images_in_image_dir: int = 5000,
    dry_run: bool = True,
):
    # create the target directories
    image_subset_dir = (
        path_to_image_subsets_dir / path_to_image_dir.parent.name / "images"
    )
    label_subset_dir = (
        path_to_image_subsets_dir / path_to_image_dir.parent.name / "labels"
    )
    image_subset_dir.mkdir(parents=True, exist_ok=True)
    label_subset_dir.mkdir(parents=True, exist_ok=True)

    source_imgs = list(path_to_image_dir.glob("*.png"))
    if len(source_imgs) < min_num_images_in_image_dir:
        print(
            f"{len(source_imgs)} imgs in {path_to_image_dir} (need at least {min_num_images_in_image_dir})"
        )
        return

    source_imgs = [
        img_path for img_path in source_imgs if does_the_path_match(regex_str, img_path)
    ]

    for source_img in source_imgs:
        source_label = (
            source_img.parent.parent / "labels" / source_img.with_suffix(".txt").name
        )
        if dry_run:
            print(
                f"[DRY] shutil.move({source_img}, {image_subset_dir / source_img.name})"
            )
            print(
                f"[DRY] shutil.move({source_label}, {label_subset_dir / source_label.name})"
            )
        else:
            shutil.move(source_img, image_subset_dir / source_img.name)
            try:
                shutil.move(source_label, label_subset_dir / source_label.name)
            except FileNotFoundError:
                print(f"could not find label file {source_label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "move subsets of images from `images` folders to a folder in a new directory"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--path-to-runset", type=Path, help="path to run folders")
    group.add_argument(
        "--from-list",
        type=Path,
        help=(
            "a text file where each line is the path to one run that "
            "will be processed. note that this is the path to the run "
            "folder, which is the folder that has the `images` folder "
            "and optionally the `labels` folder"
        ),
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help=(
            "image directory for the folders of images - the folder structure "
            "for path_to_runsets will be mimicked"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if set, will not move any files, just print the commands",
    )

    args = parser.parse_args()

    if args.path_to_runset:
        run_set = args.path_to_runset
        if not run_set.exists():
            raise FileNotFoundError(f"directory {run_set} not found")

        folders = sorted(
            [file for file in Path(run_set).rglob("images") if is_not_hidden_path(file)]
        )

        if len(folders) == 0:
            raise ValueError(f"no image folders found in directory {run_set}")

        multiprocess_fn(
            folders,
            partial(
                move_images,
                path_to_image_subsets_dir=args.image_dir,
                regex_str=r"img_00\d\d\d.png",
                dry_run=args.dry_run,
            ),
            ordered=False,
        )
    elif args.from_list:
        run_list = args.from_list
        if not run_list.exists():
            raise FileNotFoundError(f"directory {run_list} not found")

        with open(run_list) as f:
            given_folders = [
                Path(p) / "images"
                for p in f.read().splitlines()
                if (Path(p) / "images").exists()
            ]

        if len(given_folders) == 0:
            raise ValueError(f"no paths given in file {run_list}")

        multithread_map_unordered(
            given_folders,
            partial(
                move_images,
                path_to_image_subsets_dir=args.image_dir,
                regex_str=r"img_00\d\d\d.png",
                dry_run=args.dry_run,
            ),
        )
    else:
        raise ValueError("This shouldn't be possible!")
