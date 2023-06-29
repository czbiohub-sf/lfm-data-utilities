#! /usr/bin/env python3


from pathlib import Path
from typing import List
from urllib.request import pathname2url
from functools import partial

from lfm_data_utilities.malaria_labelling.labelling_constants import (
    IMG_WIDTH,
    IMG_HEIGHT,
    IMAGE_SERVER_PORT,
)
from lfm_data_utilities.utils import (
    multiprocess_fn,
    path_relative_to,
)


from lfm_data_utilities.malaria_labelling.label_studio_converter.yogo_format_converter import convert_yolo_to_ls


PARASITE_DATA_RUNSET_PATH = Path(
    "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/"
)


def generate_tasks_for_runset_by_parent_folder(
    path_to_runset_folder: Path,
    path_to_parent_for_image_server: Path,
    label_dir_name="labels",
    tasks_file_name="tasks",
):
    folders = [
        Path(p).parent for p in path_to_runset_folder.glob(f"./**/{label_dir_name}")
    ]

    if len(folders) == 0:
        raise ValueError(
            "couldn't find labels and images - double check the provided path"
        )
    multiprocess_fn(
        folders,
        partial(
            gen_task,
            relative_parent=path_to_parent_for_image_server,
            label_dir_name=label_dir_name,
            tasks_file_name=tasks_file_name,
        ),
        ordered=False,
    )


def generate_tasks_for_runset(
    run_folders: List[Path],
    relative_parent: Path,
    label_dir_name="labels",
    tasks_file_name="tasks",
    use_tqdm=False,
):
    multiprocess_fn(
        run_folders,
        partial(
            gen_task,
            relative_parent=relative_parent,
            label_dir_name=label_dir_name,
            tasks_file_name=tasks_file_name,
        ),
        ordered=False,
        verbose=use_tqdm,
    )


def gen_task(
    folder_path: Path,
    relative_parent: Path,
    label_dir_name="labels",
    tasks_file_name="tasks",
):
    abbreviated_path = str(path_relative_to(folder_path, relative_parent))
    root_url = (
        f"http://localhost:{IMAGE_SERVER_PORT}/{pathname2url(abbreviated_path)}/images"
    )

    tasks_path = str(folder_path / Path(tasks_file_name).with_suffix(".json"))

    try:
        convert_yolo_to_ls(
            input_dir=str(folder_path),
            out_file=tasks_path,
            label_dir_name=label_dir_name,
            out_type="predictions",
            image_root_url=root_url,
            image_ext=".png",
            image_dims=(IMG_WIDTH, IMG_HEIGHT),
            ignore_images_without_labels=True,
        )
    except TypeError:
        # we aren't using our custom version, so try default
        print(
            "warning: couldn't give convert_yolo_to_ls image dims, so defaulting "
            "to slow version. Will import"
        )
        convert_yolo_to_ls(
            input_dir=str(folder_path),
            out_file=tasks_path,
            out_type="predictions",
            image_root_url=root_url,
            image_ext=".png",
        )
    except Exception as e:
        print(f"exception found for file {folder_path}: {e}. continuing...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("label a set of run folders")
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument(
        "--label-dir-name",
        default="labels",
        help="name for label dir for each runset - defaults to 'labels'",
    )
    parser.add_argument(
        "--tasks-file-name",
        default="tasks",
        help="name for label studio tasks file - defaults to tasks.json",
    )

    args = parser.parse_args()
    path_to_runset = args.path_to_runset

    if not path_to_runset.exists():
        raise ValueError(f"{str(path_to_runset)} doesn't exist")

    generate_tasks_for_runset_by_parent_folder(
        path_to_runset,
        PARASITE_DATA_RUNSET_PATH,
        label_dir_name=args.label_dir_name,
        tasks_file_name=args.tasks_file_name,
    )
