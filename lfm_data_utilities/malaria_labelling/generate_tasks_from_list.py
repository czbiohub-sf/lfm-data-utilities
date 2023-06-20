#! /usr/bin/env python3


"""
This script generates a folder + tasks from a list of images and labels.

The challenging part of this problem is keeping the map from label to
it's original location, so we can sort the labels back to the correct
locations.
"""

import math
import shutil
import argparse

from pathlib import Path
from typing import List, Tuple, Dict

from lfm_data_utilities.utils import PathLike


def make_yogo_label_dir(
    dir_name: str, image_label_pairs: List[Tuple[PathLike, PathLike]]
):
    """
    This function takes a list of image label pairs and creates a directory with name
    dir_name. It will have `images` and `labels` subdirectories, and will copy the
    images and labels to those subdirectories.

    dir_name:
        Name of the directory to create.
    image_label_pairs:
        List of (path to image, path to label) pairs. We will rename the image and labels
        to a common name, and then create a mapping from the common name to the original
        image and label. this mapping is saved as "image_label_map.txt" in the directory.
    """
    out_dir = Path(dir_name)
    out_dir.mkdir(exist_ok=True, parents=True)

    image_dir = out_dir / "images"
    image_dir.mkdir(exist_ok=True, parents=True)

    label_dir = out_dir / "labels"
    label_dir.mkdir(exist_ok=True, parents=True)

    filename_map: Dict[str, str] = {}

    N = int(math.log(len(image_label_pairs), 10) + 1)
    for i, (image_path, label_path) in enumerate(image_label_pairs):
        image_name = f"rand_image_{i:0{N}}.png"
        label_name = f"rand_label_{i:0{N}}.txt"

        try:
            shutil.copy(image_path, image_dir / image_name)
            shutil.copy(label_path, label_dir / label_name)
        except FileNotFoundError:
            print(f"Could not find {image_path} or {label_path}")
            continue

        filename_map[image_name] = str(
            image_path.resolve() if isinstance(image_path, Path) else image_path
        )
        filename_map[label_name] = str(
            label_path.resolve() if isinstance(label_path, Path) else label_path
        )

    with open(out_dir / "image_label_map.txt", "w") as f:
        for k, v in filename_map.items():
            f.write(f"{k} {v}\n")


def sort_corrected_labels(corrected_label_dir, filename_map_path):
    """
    This function takes a directory with corrected labels and the original source
    and sorts the corrected labels into the same order as the source.
    """
    with open(filename_map_path, "r") as f:
        filename_map = dict([line.split() for line in f.readlines()])

    for filename, source in filename_map.items():
        shutil.copy(corrected_label_dir / filename, Path(source).parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a folder with images and labels from a list of images and labels",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(
        help="correct or re-sort? it is up to you!", dest="task"
    )

    # TODO: implement! for now, raise NotImplementedError
    correct_parser = subparsers.add_parser(
        "correct", help="correct labels", allow_abbrev=False
    )

    resort = subparsers.add_parser("resort", help="re-sort labels", allow_abbrev=False)
    resort.add_argument(
        "corrected_label_dir", type=Path, help="path to corrected label dir"
    )
    resort.add_argument("filename_map_path", type=Path, help="path to filename map (i.e. from the ")

    args = parser.parse_args()

    if args.task == "correct":
        raise NotImplementedError(
            "Correcting labels is not yet implemented - call from "
            "a python program with your list of images and labels"
        )
    elif args.task == "resort":
        sort_corrected_labels(args.corrected_label_dir, args.filename_map_path)
    else:
        args.print_help()
