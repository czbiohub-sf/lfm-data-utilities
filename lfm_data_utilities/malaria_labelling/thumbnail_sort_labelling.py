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


import shutil
import argparse

from tqdm import tqdm
from typing import List
from pathlib import Path

from yogo.data.dataset import YOGO_CLASS_ORDERING
from yogo.data.dataset_description_file import load_dataset_description
from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import gen_task


def create_folders_for_output_dir(
    output_dir_path: Path, classes: List[str], force_overwrite: bool = False
):
    """creates the 'thumbnail-folder' above"""
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


def create_tasks_files_for_run_sets(
    path_to_labelled_data_ddf: Path, label_dir_name: str
):
    ddf = load_dataset_description(path_to_labelled_data_ddf)
    dataset_paths = ddf.dataset_paths + (ddf.test_dataset_paths or [])

    for d in tqdm(dataset_paths):
        image_path = d["image_path"]
        label_path = d["label_path"]
        gen_task(
            folder_path=Path(label_path).parent,
            images_dir_path=image_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_output_dir", type=Path)
    parser.add_argument(
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
    parser.add_argument(
        "--label-dir-name",
        help=(
            "name of the directory containing the label files - defaults to 'labels'. "
            "in general you should not need to change this, "
        ),
        default="labels",
    )
    parser.add_argument(
        "--overwrite-previous-thumbnails",
        action="store_true",
        help="if set, will overwrite previous thumbnails",
    )

    args = parser.parse_args()

    create_folders_for_output_dir(
        args.path_to_output_dir,
        YOGO_CLASS_ORDERING,
        force_overwrite=args.overwrite_previous_thumbnails,
    )
    task_files = create_tasks_files_for_run_sets(
        args.path_to_labelled_data_ddf, label_dir_name=args.label_dir_name
    )
