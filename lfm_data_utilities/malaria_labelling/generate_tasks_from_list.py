#! /usr/bin/env python3


"""
This script generates a folder + tasks from a list of images and labels.

The challenging part of this problem is keeping the map from label to
it's original location, so we can sort the labels back to the correct
locations.
"""

import json
import math
import shutil
import argparse

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict

from yogo.data.dataset import YOGO_CLASS_ORDERING

from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import (
    generate_tasks_for_runset,
)

from lfm_data_utilities.utils import PathLike


# label studio chose a silly way to represent data
# why not make categories a map from id to name or vice-versa?
# instead we have to iterate through categories by id or name to
# get the class we want. Why a list of dicts?? whyyy!!!???
MASTER_NOTES_DOT_JSON = {
    "categories": [
        {"id": "1", "name": "healthy"},
        {"id": "2", "name": "ring"},
        {"id": "3", "name": "trophozoite"},
        {"id": "4", "name": "schizont"},
        {"id": "5", "name": "gametocyte"},
        {"id": "6", "name": "wbc"},
        {"id": "7", "name": "misc"},
    ],
    "info": {"year": 2023, "version": "1.0", "contributor": "Label Studio"},
}


MASTER_NAME_TO_ID = dict(
    [(row["name"], row["id"]) for row in MASTER_NOTES_DOT_JSON["categories"]]  # type: ignore
)


def copy_label(label_path: Path, output_path):
    """Copies label_path, aware of variance of idx-to-label mapping done by label-studio"""
    with open(label_path.parent.parent / "notes.json", "r") as f:
        label_notes_json = json.load(f)
        label_id_to_class_name = dict(
            [(row["id"], row["name"]) for row in label_notes_json["categories"]]  # type: ignore
        )

    with open(label_path, "r") as f:
        label_data = f.readlines()

    bboxes = []
    for row in label_data:
        row_numbers = row.split(" ")
        row_class = label_id_to_class_name[row_numbers[0]]
        row_corrected_label = MASTER_NAME_TO_ID[row_class]
        bboxes.append(" ".join(row_corrected_label + row_numbers[1:]))

    with open(output_path, "w") as f:
        f.write("\n".join(bboxes))


def make_yogo_label_dir(
    dir_name: str, image_label_pairs: List[Tuple[PathLike, PathLike]]
):
    """
    This function takes a list of image label pairs and creates a directory with name
    dir_name. It will have `images` and `labels` subdirectories, and will copy the
    images and labels to those subdirectories.

    A major annoying problem is that there is no garuntee that labels from different
    label-studio runs have the same ordering, so we need to bring the individual
    notes.json files over somehow.

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
    for i, (image_path, label_path) in enumerate(tqdm(image_label_pairs)):
        image_name = f"rand_image_{i:0{N}}.png"
        label_name = f"rand_image_{i:0{N}}.txt"

        try:
            shutil.copy(image_path, image_dir / image_name)
            copy_label(label_path, label_dir / label_name)
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

    with open(out_dir / "classes.txt", "w") as f:
        for c in YOGO_CLASS_ORDERING:
            f.write(f"{c}\n")

    generate_tasks_for_runset(
        [out_dir],
        out_dir,
        label_dir_name=label_dir,
        tasks_file_name="tasks.json",
        use_tqdm=True,
    )


def sort_corrected_labels(corrected_label_dir, filename_map_path):
    """
    This function takes a directory with corrected labels and the original source
    and sorts the corrected labels into the same order as the source.
    """
    with open(filename_map_path, "r") as f:
        filename_map = dict([line.split() for line in f.readlines()])

    # TODO need to make it more "interactive" - smth like a pre-commit stage
    # that displays the copies that *will* be made
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
    resort.add_argument(
        "filename_map_path", type=Path, help="path to filename map (i.e. from the "
    )

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