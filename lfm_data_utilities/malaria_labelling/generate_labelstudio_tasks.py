#! /usr/bin/env python3


from tqdm import tqdm
from pathlib import Path
from typing import Union, List
from urllib.request import pathname2url
from functools import partial

from lfm_data_utilities.utils import multiprocess_fn

from labelling_constants import IMG_WIDTH, IMG_HEIGHT, IMAGE_SERVER_PORT

from label_studio_converter.imports.yolo import convert_yolo_to_ls


def path_is_relative_to(path_a: Path, path_b: Union[str, Path]) -> bool:
    """
    Path.is_relative_to is available in pathlib since 3.9,
    but we are running 3.7. Copied from pathlib
    (https://github.com/python/cpython/blob/main/Lib/pathlib.py)
    """
    path_b = type(path_a)(path_b)
    return path_a == path_b or path_b in path_a.parents


def path_relative_to(path_a: Path, path_b: Union[str, Path], walk_up=False) -> Path:
    """
    Path.relative_to is available in pathlib since 3.9,
    but we are running 3.7. Copied from pathlib
    (https://github.com/python/cpython/blob/main/Lib/pathlib.py)
    """
    path_cls = type(path_a)
    path_b = path_cls(path_b)

    for step, path in enumerate([path_b] + list(path_b.parents)):
        if path_is_relative_to(path_a, path):
            break
    else:
        raise ValueError(f"{str(path_a)!r} and {str(path_b)!r} have different anchors")

    if step and not walk_up:
        raise ValueError(f"{str(path_a)!r} is not in the subpath of {str(path_b)!r}")

    parts = ("..",) * step + path_a.parts[len(path.parts) :]
    return path_cls(*parts)


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
    if use_tqdm:
        tqdm_ = tqdm
    else:

        def tqdm_(v):
            return v

    multiprocess_fn(
        run_folders,
        partial(
            gen_task,
            relative_parent=relative_parent,
            label_dir_name=label_dir_name,
            tasks_file_name=tasks_file_name,
        ),
        ordered=False,
    )


def gen_task(
    folder_path: Path,
    relative_parent: Path,
    label_dir_name="labels",
    tasks_file_name="tasks",
):
    abbreviated_path = str(path_relative_to(folder_path, relative_parent))
    print(f"ABBREV {abbreviated_path}")
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
        )
    except TypeError:
        # we aren't using our custom version, so try default
        print(
            "warning: couldn't give convert_yolo_to_ls image dims, so defaulting "
            "to slow version"
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

    # TODO maybe this path_to_runset folder should just be flexo?
    parasite_data_runset_path = Path(
        "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/"
    )
    generate_tasks_for_runset_by_parent_folder(
        path_to_runset,
        parasite_data_runset_path,
        label_dir_name=args.label_dir_name,
        tasks_file_name=args.tasks_file_name,
    )
