#! /usr/bin/env python3

""" Thumbnail Sort Labelling

High-level Goal:
- take path to folder of YOLO style labels
- create `tasks.json` file (files?)
- create folders of thumbnails (the 'thumbnail-folder'), sorted by class
    - thumbnail filenames should include class, run id, cell id
    - should also create 'target' folders in the same place!
    - create a file in the thumbnail folder that maps a run id to the path of the tasks.json file
- once sorted files are created,
    - go through folders, correcting labels in the tasks.json file(s?)
- re-export the tasks.json file(s)

Notes:
- once a run has cell classes that are overwritten, we can NOT re-export
that run from label studio, since it will no longer have correct files.
"""


import shutil
import argparse

from typing import List
from pathlib import Path

from yogo.data.dataset import YOGO_CLASS_ORDERING
from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import (
    generate_tasks_for_runset,
    PARASITE_DATA_RUNSET_PATH,
)


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


def create_tasks_files_for_run_sets(path_to_run_sets: Path, label_dir_name: str):
    paths_to_run_sets = [p.parent for p in path_to_run_sets.glob("./**/labels")]

    generate_tasks_for_runset(
        run_folders=paths_to_run_sets,
        relative_parent=PARASITE_DATA_RUNSET_PATH,
        label_dir_name=label_dir_name,
        tasks_file_name="tasks_for_thumbnails.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_output_dir", type=Path)
    parser.add_argument(
        "--path-to-run-sets",
        help=(
            "path to folder of run sets - in general you should not need to change this, "
            "since we mostly want to correct labels for human-labelled data (i.e. biohub-labels/vetted)"
        ),
        default=Path(
            "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels/vetted"
        ),
        type=Path,
    )
    parser.add_argument(
        "--label-dir-name",
        default="labels",
        help="name for label dir for each runset",
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
        args.path_to_run_sets, label_dir_name=args.label_dir_name
    )
