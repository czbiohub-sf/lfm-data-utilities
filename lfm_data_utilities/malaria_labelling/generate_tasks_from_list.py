#! /usr/bin/env python3


"""
This script generates a folder + tasks from a list of images and labels.

The challenging part of this problem is keeping the map from label to
it's original location, so we can sort the labels back to the correct
locations.

NOTE (13/07/2023)

I think this method is probably not great; it's a bit too complicated to justify
how low throughput label-studio is - for that reason, I am depricating this (but
keeping it around just in case).
"""

import json
import math
import shutil
import argparse
import warnings

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

from yogo.data import YOGO_CLASS_ORDERING


from lfm_data_utilities.malaria_labelling.labelling_constants import IMG_SERVER_ROOT
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
        {"id": "0", "name": "healthy"},
        {"id": "1", "name": "ring"},
        {"id": "2", "name": "trophozoite"},
        {"id": "3", "name": "schizont"},
        {"id": "4", "name": "gametocyte"},
        {"id": "5", "name": "wbc"},
        {"id": "6", "name": "misc"},
    ],
    "info": {"year": 2023, "version": "1.0", "contributor": "Label Studio"},
}


MASTER_NAME_TO_ID = dict(
    [(row["name"], row["id"]) for row in MASTER_NOTES_DOT_JSON["categories"]]  # type: ignore
)

MASTER_ID_TO_NAME = {v: k for k, v in MASTER_NAME_TO_ID.items()}


def copy_label_to_central_dir(label_path: Path, output_path: Path):
    """Copies label_path into central dir, aware of variance of idx-to-label mapping done by label-studio"""
    # if we haven't exported from label studio, there will not be a notes.json. The files will be in
    # the MASTER_NOTES_DOT_JSON format
    try:
        with open(label_path.parent.parent / "notes.json", "r") as f:
            label_notes_json = json.load(f)
            label_id_to_class_name = dict(
                [(str(row["id"]), row["name"]) for row in label_notes_json["categories"]]  # type: ignore
            )
    except FileNotFoundError:
        label_id_to_class_name = MASTER_ID_TO_NAME

    with open(label_path, "r") as f:
        label_data = f.read().strip().split("\n")

    bboxes = []
    for row in label_data:
        row_numbers = row.split(" ")
        row_class = label_id_to_class_name[row_numbers[0]]
        row_corrected_label = MASTER_NAME_TO_ID[row_class]
        bboxes.append(" ".join([row_corrected_label, *row_numbers[1:]]))

    with open(output_path, "w") as f:
        out_data = "\n".join(bboxes).strip()
        f.write(out_data)


def copy_label_to_original_dir(label_path: Path, output_path: Path):
    """Copies label_path back into original dir, aware of variance of idx-to-label mapping done by label-studio"""
    if not label_path.exists():
        warnings.warn(f"{label_path} does not exist; perhaps the labeller deleted it?")
        return

    with open(label_path.parent.parent / "notes.json", "r") as f:
        label_notes_json = json.load(f)
        source_id_to_name = dict(
            [(row["id"], str(row["name"])) for row in label_notes_json["categories"]]  # type: ignore
        )

    try:
        with open(output_path.parent.parent / "notes.json", "r") as f:
            label_notes_json = json.load(f)
            label_name_to_id = dict(
                [(row["name"], str(row["id"])) for row in label_notes_json["categories"]]  # type: ignore
            )
    except FileNotFoundError:
        label_name_to_id = MASTER_NAME_TO_ID

    with open(label_path, "r") as f:
        label_data = f.read().strip().split("\n")

    bboxes = []
    for row in label_data:
        row_numbers = row.split(" ")
        row_class = source_id_to_name[row_numbers[0]]
        row_corrected_label = label_name_to_id[row_class]
        bboxes.append(" ".join([row_corrected_label, *row_numbers[1:]]))

    with open(output_path, "w") as f:
        out_data = "\n".join(bboxes).strip()
        f.write(out_data)


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

    label_name_map: Dict[str, str] = {}
    image_name_map: Dict[str, str] = {}

    N = int(math.log(len(image_label_pairs), 10) + 1)
    for i, (image_path, label_path) in enumerate(tqdm(image_label_pairs)):
        image_name = f"rand_image_{i:0{N}}.png"
        label_name = f"rand_image_{i:0{N}}.txt"

        try:
            shutil.copy(image_path, image_dir / image_name)
            copy_label_to_central_dir(Path(label_path), label_dir / label_name)
        except FileNotFoundError as e:
            print(f"Could not find a file: {e}")
            continue

        label_name_map[label_name] = str(
            label_path.resolve() if isinstance(label_path, Path) else label_path
        )
        image_name_map[image_name] = str(
            image_path.resolve() if isinstance(image_path, Path) else image_name
        )

    with open(out_dir / "image_map.txt", "w") as f:
        for k, v in image_name_map.items():
            f.write(f"{k} {v}\n")

    with open(out_dir / "label_map.txt", "w") as f:
        for k, v in label_name_map.items():
            f.write(f"{k} {v}\n")

    with open(out_dir / "classes.txt", "w") as f:
        for c in YOGO_CLASS_ORDERING:
            f.write(f"{c}\n")

    generate_tasks_for_runset(
        run_folders=[out_dir],
        relative_parent=IMG_SERVER_ROOT,
        label_dir_name=label_dir,
        tasks_file_name="tasks.json",
        use_tqdm=True,
    )


def sort_corrected_labels(
    corrected_label_dir, filename_map_path, output_dir_override=None
):
    """
    This function takes a directory with corrected labels and the original source
    and sorts the corrected labels into the same order as the source.
    #
    this code should be considered harmful
    """
    with open(filename_map_path, "r") as f:
        filename_map = dict([line.split() for line in f.readlines()])

    d = defaultdict(list)
    for corrected_file_name, original_file_name in filename_map.items():
        run_name = Path(original_file_name).parent.parent.name
        d[run_name].append((corrected_file_name, original_file_name))

    if output_dir_override is not None:
        for k in d:
            (output_dir_override / k).mkdir(parents=True)
            (output_dir_override / k / "labels").mkdir(parents=True)
            (output_dir_override / k / "images").mkdir(parents=True)
            with open((output_dir_override / k) / "classes.txt", "w") as f:
                for c in YOGO_CLASS_ORDERING:
                    f.write(f"{c}\n")
            shutil.copy(
                corrected_label_dir / "notes.json",
                output_dir_override / k / "notes.json",
            )

            for corrected_file_name, original_file_name in d[k]:
                # copy it over
                # it is either a text file or png; if text file, put it
                # in labels; if png, put it in images :)
                assert Path(
                    original_file_name
                ).exists(), f"{original_file_name} doesn't exist! {Path(original_file_name).exists()}"

                if corrected_file_name.endswith(".txt"):
                    label_path = (
                        Path(corrected_label_dir) / "labels" / corrected_file_name
                    )
                    assert (
                        label_path.exists()
                    ), f'{(Path(corrected_label_dir) / "labels" / corrected_file_name)} doesnt exist!'
                    shutil.copy(
                        label_path,
                        output_dir_override
                        / k
                        / "labels"
                        / Path(original_file_name).with_suffix(".txt").name,
                    )
                elif corrected_file_name.endswith(".png"):
                    shutil.copy(
                        original_file_name, output_dir_override / k / "images",
                    )


#     # TODO need to make it more "interactive" - smth like a pre-commit stage
#     # that displays the copies that *will* be made
#     for filename, source in filename_map.items():
#         copy_label_to_original_dir(
#             corrected_label_dir / "labels" / filename, Path(source)
#         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a folder with images and labels from a list of images and labels",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(
        help="correct or re-sort? it is up to you!", dest="task"
    )

    correct_parser = subparsers.add_parser(
        "correct", help="correct labels", allow_abbrev=False
    )

    resort_parser = subparsers.add_parser(
        "resort", help="re-sort labels", allow_abbrev=False
    )
    resort_parser.add_argument(
        "corrected_label_dir", type=Path, help="path to corrected label dir"
    )
    resort_parser.add_argument(
        "filename_map_path", type=Path, help="path to filename map (i.e. from the "
    )
    resort_parser.add_argument(
        "--output-dir-override",
        type=Path,
        help=(
            "output directory for the corrections. if not "
            "provided, corrections will replace the original file locations"
        ),
    )

    args = parser.parse_args()

    if args.task == "correct":
        raise NotImplementedError(
            "Correcting labels is not yet implemented - call from "
            "a python program with your list of images and labels"
        )
    elif args.task == "resort":
        sort_corrected_labels(
            args.corrected_label_dir, args.filename_map_path, args.output_dir_override
        )
    else:
        parser.print_help()
