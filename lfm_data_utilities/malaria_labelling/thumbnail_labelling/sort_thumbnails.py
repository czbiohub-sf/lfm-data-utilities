#! /usr/bin/env python3

import json
import shutil

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, DefaultDict, cast

from yogo.data import YOGO_CLASS_ORDERING

from lfm_data_utilities.utils import timing_context_manager
from lfm_data_utilities.malaria_labelling.labelling_constants import CLASSES
from lfm_data_utilities.malaria_labelling.label_studio_converter.convert_ls_to_yolo import (
    convert_ls_to_yolo,
)


DEFAULT_LABELS_PATH = Path(
    "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels/"
)


def parse_thumbnail_name(thumbnail_name: str) -> Tuple[str, str, str]:
    """
    parses a thumbnail name into class, cell_id, and task.json id

    We can just remove the '.png' and split on '_', since class, cell_id, and task_json_id don't
    have underscores in them.
    """
    t = tuple(s.strip() for s in thumbnail_name.replace(".png", "").split("_"))
    if len(t) != 3:
        raise ValueError(f"invalid thumbnail name {thumbnail_name}")
    return cast(Tuple[str, str, str], t)


def backup_vetted(commit: bool = True, _backup: bool = True):
    with timing_context_manager(f"creating backup of {DEFAULT_LABELS_PATH / 'vetted'}"):
        if commit and _backup:
            vetted_backup_path = str(
                DEFAULT_LABELS_PATH
                / "vetted-backup"
                / f"backup-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            )
            shutil.make_archive(
                vetted_backup_path, "zip", DEFAULT_LABELS_PATH / "vetted"
            )


def get_list_of_corrections(
    path_to_thumbnails: Path,
) -> DefaultDict[str, List[Dict[str, str]]]:
    id_to_list_of_corrections: DefaultDict[str, List[Dict[str, str]]] = defaultdict(
        list
    )
    for class_ in YOGO_CLASS_ORDERING:
        corrected_class_dir = path_to_thumbnails / f"corrected_{class_}"
        if not corrected_class_dir.exists():
            # user ignored this class, so just skipp it
            continue

        for thumbnail in corrected_class_dir.iterdir():
            original_class, cell_id, task_json_id = parse_thumbnail_name(thumbnail.name)

            id_to_list_of_corrections[task_json_id].append(
                {
                    "cell_id": cell_id,
                    "original_class": original_class,
                    "corrected_class": class_,
                }
            )
    return id_to_list_of_corrections


def find_cell_indices_id_map(tasks: Dict) -> Dict[str, Dict[str, int]]:
    """ for each cell in the tasks.json dict, find it's image index and bbox index

    The idea is something like this

    >>> indexes_by_id = find_cell_indices_id_map(tasks)
    >>> for correction in corrections:
    ...     image_index = indexes_by_id[correction["id"]]["image_index"]
    ...     bbox_index = indexes_by_id[correction["id"]]["bbox_index"]
    ...     tasks \
    ...         [image_index] \
    ...         ["predictions"][0]["result"] \
    ...         [bbox_index] \
    ...         ["value"]["rectanglelabels"] \
    ...         = [correction["corrected_class"]]
    """
    d = dict()
    for i, image_prediction in enumerate(tasks):
        bbox_predictions = image_prediction["predictions"][0]["result"]  # lol
        for j, bbox_prediction in enumerate(bbox_predictions):
            d[bbox_prediction["id"]] = {"image_index": i, "bbox_index": j}
    return d


def sort_thumbnails(path_to_thumbnails: Path, commit=True, _backup=False):
    """
    The thumbnails dir should have three things:
        - a set of folders named after the classes ("Class Folder" from now on)
        - a set of folders named "corrected_<class>" for each class ("Corrected Class Folder" from now on)
        - a file named "id_to_task_path.json" which maps task ids to task.json files

    After creation, each thumbnail has the name <class>_<cell_id>_<id>.png. `id` is the id of the task.json file,
    which one can get from id_to_task_path.json. `cell_id` is the id of the cell in that tasks.json. `class` is
    the cell's original class.

    The labeller will sort the thumbnails into the "Corrected Class Folder" for the class that they deem is the
    true class for that cell.

    This function will first create a backup of `LFM_scope/biohub-labels/vetted` into `LFM_scope/biohub-labels/vetted-backup`.
    Then, it goes through the corrected class folders and will correct the tasks.json files. It will then export the

    TODO hard coding DEFAULT_LABELS_PATH. What should we do with it? All the tasks.json paths should be from there, so this
    information is redundant. Maybe we will use this to verify that the tasks.json files are correct? Need to reconsider this.
    """
    with open(path_to_thumbnails / "id_to_task_path.json") as f:
        id_to_task_path = json.load(f)
        id_to_task_path = cast(Dict[str, Dict[str, str]], id_to_task_path)

    for label_and_task_path in id_to_task_path.values():
        task_path = path_to_thumbnails / "tasks" / label_and_task_path["task_name"]
        if not task_path.exists():
            raise ValueError(f"task_path {task_path} does not exist")

    # create backup of vetted
    backup_vetted(commit=commit, _backup=_backup)

    # create a list of all the corrections
    id_to_list_of_corrections = get_list_of_corrections(path_to_thumbnails)

    # Iterate through the corrections tasks-wise
    # This is going to be horifically inefficient - label studio chose their
    # tasks.json format poorly. They have list of dicts of predictions, and each
    # dict has an id. Why not make it a dict of dicts, mapping the cell id to the
    # prediction? It would turn the cell search from O(n) to O(1). Oh well, do a
    # first pass over the dict, making a map of cell id to image index and bbox index
    not_corrected = was_corrected = 0
    for task_json_id, corrections in id_to_list_of_corrections.items():
        if len(corrections) == 0:
            continue

        # read the json file
        task_path = path_to_thumbnails / "tasks" / label_and_task_path["task_name"]
        with open(task_path) as f:
            tasks = json.load(f)

        indexes_by_id = find_cell_indices_id_map(tasks)

        for correction in corrections:
            cell_id = correction["cell_id"]
            original_class = correction["original_class"]
            correction["corrected_class"]

            try:
                image_index = indexes_by_id[cell_id]["image_index"]
                bbox_index = indexes_by_id[cell_id]["bbox_index"]
            except KeyError:
                not_corrected += 1
                print(
                    f"could not find cell_id {cell_id} in task {id_to_task_path[task_json_id]}"
                )
                continue

            bbox_pred = tasks[image_index]["predictions"][0]["result"][bbox_index]

            id_is_correct = bbox_pred["id"] == cell_id
            original_class_matches = bbox_pred["value"]["rectanglelabels"] == [
                original_class
            ]
            if not (id_is_correct and original_class_matches):
                not_corrected += 1
                print(
                    f"cell_id {cell_id} does not match bbox_pred id {bbox_pred['id']} or "
                    f"original_class {original_class} does not match bbox_pred class "
                    f"{bbox_pred['value']['rectanglelabels']}"
                )
                continue

            bbox_pred["value"]["rectanglelabels"] = [correction["corrected_class"]]

            was_corrected += 1

        # write the (corrected) json file
        if commit:
            with open(task_path, "w") as f:
                json.dump(tasks, f)
        else:
            print("would have written corrected tasks.json file to {task_path}")

    print(f"not corrected: {not_corrected}, was corrected: {was_corrected}")

    # convert the corrected json files to yolo format
    for task_and_label_path in id_to_task_path.values():
        task_path = path_to_thumbnails / "tasks" / task_and_label_path["task_name"]
        label_path = Path(task_and_label_path["label_path"])
        if commit:
            convert_ls_to_yolo(
                path_to_ls_file=task_path,
                path_to_output_dir=label_path.parent.resolve(),
                classes=CLASSES,
                overwrite_existing_labels=commit,
                download_images=False,
            )
        else:
            print(f"would have overwritten YOGO labels at {task_path.parent}")
