#! /usr/bin/env python3

""" Thumbnail Sort Labelling

High-level Goal:
- take dataset description file for
- create `tasks.json` file (files?)
- create folders of thumbnails (the 'thumbnail-folder'), sorted by class
    - thumbnail filenames should include class, run id, cell id
    - should also create 'target' folders in the same place!
    - create a file in the thumbnail folder that maps a run id to the path of the tasks.json file
- once sorted files are created,
    - go through folders, correcting labels in the tasks.json file(s?)
- zip and save biohub-labels/vetted as a backup
- re-export the tasks.json file(s)

Notes:
- once a run has cell classes that are overwritten, we can NOT re-export
that run from label studio, since it will no longer have correct files.
"""

import math
import json
import shutil
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional

from yogo.data.dataset import YOGO_CLASS_ORDERING
from yogo.data.dataset_description_file import load_dataset_description
from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import (
    gen_task,
    LFM_SCOPE_PATH,
)


def create_folders_for_output_dir(
    output_dir_path: Path, classes: List[str], force_overwrite: bool = False
) -> Dict[str, Path]:
    """creates the 'thumbnail-folder' above"""
    class_dirs = {}
    for class_ in classes:
        class_dir = output_dir_path / class_
        corrected_class_dir = output_dir_path / f"corrected_{class_}"

        if force_overwrite:
            if class_dir.exists():
                shutil.rmtree(class_dir)
            if corrected_class_dir.exists():
                shutil.rmtree(corrected_class_dir)

        class_dir.mkdir(exist_ok=True, parents=True)
        corrected_class_dir.mkdir(exist_ok=True, parents=True)

        class_dirs[class_] = class_dir
    return class_dirs


def create_tasks_files_for_run_sets(
    path_to_labelled_data_ddf: Path, label_dir_name: str
) -> List[Path]:
    ddf = load_dataset_description(path_to_labelled_data_ddf)
    dataset_paths = ddf.dataset_paths + (ddf.test_dataset_paths or [])

    task_paths = []
    for d in tqdm(dataset_paths):
        image_path = d["image_path"]
        label_path = d["label_path"]
        task_file = gen_task(
            folder_path=Path(label_path).parent,
            images_dir_path=image_path,
        )
        task_paths.append(task_file)
    return task_paths


def create_thumbnails_from_tasks_and_images(
    tasks_json_path: Path, class_dirs: Dict[str, Path], run_name: Optional[str] = None
):
    run_name = run_name or tasks_json_path.parent.name

    with open(tasks_json_path) as f:
        tasks = json.load(f)

    for task in tasks:
        image_url = task["data"]["image"]

        # task.json files hold image urls that are relative to LFM_scope
        image_path = LFM_SCOPE_PATH / image_url.replace("http://localhost:8081/", "")
        image = np.array(Image.open(image_path).convert("L"))

        img_h, img_w = image.shape

        for prediction in task["predictions"][0]["result"]:
            cell_id = prediction["id"]
            class_ = prediction["value"]["rectanglelabels"][0]
            class_dir = class_dirs[class_]

            x1 = prediction["value"]["x"] / 100
            y1 = prediction["value"]["y"] / 100
            w = prediction["value"]["width"] / 100
            h = prediction["value"]["height"] / 100

            x1 = max(round(x1 * img_w), 0)
            y1 = max(round(y1 * img_h), 0)
            x2 = min(round(x1 + w * img_w), img_w - 1)
            y2 = min(round(y1 + h * img_h), img_h - 1)

            if x1 == x2 or y1 == y2:
                continue

            cell_image = image[y1:y2, x1:x2]
            pil_cell_image = Image.fromarray(cell_image)
            pil_cell_image.save(class_dir / f"{class_}_{cell_id}_{run_name}.png")


def create_thumbnails_for_sorting(
    path_to_output_dir: Path,
    path_to_labelled_data_ddf: Path,
    label_dir_name: str = "labels",
    overwrite_previous_thumbnails: bool = False,
):
    class_dirs = create_folders_for_output_dir(
        path_to_output_dir,
        YOGO_CLASS_ORDERING,
        force_overwrite=overwrite_previous_thumbnails,
    )
    task_paths = create_tasks_files_for_run_sets(
        path_to_labelled_data_ddf, label_dir_name=label_dir_name
    )

    N = int(math.log(len(task_paths), 10)) + 1
    id_to_task_path: Dict[str, str] = {}
    for i, task_path in tqdm(enumerate(task_paths), desc="creating thumbnails"):
        create_thumbnails_from_tasks_and_images(
            task_path, class_dirs, run_name=f"{i:0{N}}"
        )
        id_to_task_path[f"{i:0{N}}"] = str(task_path)

    with open(args.path_to_output_dir / "id_to_task_path.json", "w") as f:
        json.dump(id_to_task_path, f)


def sort_thumbnails(path_to_thumbnails: Path):
    """
    The thumbnails dir should have three things:
        - a set of folders named after the classes ("Class Folder" from now on)
        - a set of folders named "corrected_<class>" for each class ("Corrected Class Folder" from now on)
        - a file named "id_to_task_path.json" which maps task ids to task.json files

    After creation, each thumbnail has the name <class>_<cell_id>_<id>.png. `id` is the id of the task.json file,
    which one can get from id_to_task_path.json. `cell_id` is the id of the cell in that tasks.json. `class` is
    the cell's class.
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser")

    create_thumbnails_parser = subparsers.add_parser("create-thumbnails")
    create_thumbnails_parser.add_argument("path_to_output_dir", type=Path)
    create_thumbnails_parser.add_argument(
        "--path-to-labelled-data-ddf",
        help=(
            "path to dataset descriptor file for labelled data - in general you should not need to change this, "
            "since we mostly want to correct labels for human-labelled data (i.e. biohub-labels/vetted)"
        ),
        default=Path(
            "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels/all-labelled-data-train-only.yml"
        ),
        type=Path,
    )
    create_thumbnails_parser.add_argument(
        "--label-dir-name",
        help=(
            "name of the directory containing the label files - defaults to 'labels'. "
            "in general you should not need to change this, "
        ),
        default="labels",
    )
    create_thumbnails_parser.add_argument(
        "--overwrite-previous-thumbnails",
        action="store_true",
        help="if set, will overwrite previous thumbnails",
    )

    sort_thumbnails_parser = subparsers.add_parser("sort-thumbnails")
    sort_thumbnails_parser.add_argument("path_to_thumbnails", type=Path)

    args = parser.parse_args()
    print(args)

    if args.subparser == "sort-thumbnails":
        sort_thumbnails(args.path_to_thumbnails)
    elif args.subparser == "create-thumbnails":
        create_thumbnails_for_sorting(
            args.path_to_output_dir,
            args.path_to_labelled_data_ddf,
            args.label_dir_name,
            args.overwrite_previous_thumbnails,
        )
