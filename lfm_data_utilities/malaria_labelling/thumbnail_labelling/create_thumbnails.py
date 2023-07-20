#! /usr/bin/env python3

import math
import json
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from yogo.data.dataset_description_file import load_dataset_description

from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import (
    gen_task,
    LFM_SCOPE_PATH,
)

from lfm_data_utilities.malaria_labelling.thumbnail_labelling.create_YOGO_thumbnails import (
    create_confidence_filtered_tasks_file_from_YOGO,
)

DEFAULT_LABELS_PATH = Path(
    "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels/"
)


def create_tasks_files_from_labels(
    path_to_labelled_data_ddf: Path, tasks_dir: Path
) -> List[Dict[str, Union[int, str]]]:
    ddf = load_dataset_description(path_to_labelled_data_ddf)
    dataset_paths = ddf.dataset_paths + (ddf.test_dataset_paths or [])

    task_paths: List[Dict[str, Union[int, str]]] = []
    for i, d in tqdm(enumerate(dataset_paths)):
        image_path = d["image_path"]
        label_path = d["label_path"]
        gen_task(
            folder_path=Path(label_path).parent,
            images_dir_path=image_path,
            label_dir_name=Path(label_path).name,
            tasks_path=tasks_dir / f"thumbnail_correction_task_{i}.json",
        )
        task_paths.append(
            {
                "label_path": str(label_path),
                "task_name": f"thumbnail_correction_task_{i}.json",
                "task_num": i,
            }
        )
    return task_paths


def create_confidence_filtered_tasks_from_YOGO(
    path_to_labelled_data_ddf: Path,
    tasks_dir: Path,
    path_to_pth: Path,
    obj_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    max_class_confidence_thresh: Optional[float] = None,
) -> List[Dict[str, Union[int, str]]]:
    ddf = load_dataset_description(path_to_labelled_data_ddf)
    dataset_paths = ddf.dataset_paths + (ddf.test_dataset_paths or [])

    task_paths: List[Dict[str, Union[int, str]]] = []
    for i, d in tqdm(enumerate(dataset_paths)):
        image_path = d["image_path"]
        label_path = d["label_path"]

        create_confidence_filtered_tasks_file_from_YOGO(
            path_to_pth=path_to_pth,
            path_to_images=image_path,
            output_path=tasks_dir / f"thumbnail_correction_task_{i}.json",
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
            max_class_confidence_thresh=max_class_confidence_thresh,
        )

        task_paths.append(
            {
                "label_path": str(label_path),
                "task_name": f"thumbnail_correction_task_{i}.json",
                "task_num": i,
            }
        )
    return task_paths


def create_folders_for_output_dir(
    output_dir_path: Path,
    classes: List[str],
    force_overwrite: bool = False,
    ignore_classes: List[str] = [],
) -> Tuple[Dict[str, Path], Path]:
    """creates the 'thumbnail-folder'"""
    class_dirs = {}
    for class_ in classes:
        if class_ not in ignore_classes:
            class_dir = output_dir_path / class_
            if force_overwrite:
                if class_dir.exists():
                    shutil.rmtree(class_dir)
            class_dir.mkdir(exist_ok=True, parents=True)
            class_dirs[class_] = class_dir

        corrected_class_dir = output_dir_path / f"corrected_{class_}"
        tasks_dir = output_dir_path / "tasks"

        if force_overwrite:
            if corrected_class_dir.exists():
                shutil.rmtree(corrected_class_dir)
            if tasks_dir.exists():
                shutil.rmtree(tasks_dir)

        corrected_class_dir.mkdir(exist_ok=True, parents=True)
        tasks_dir.mkdir(exist_ok=True, parents=True)

    return class_dirs, tasks_dir


def create_thumbnail_name(class_: str, cell_id: str, task_json_id: str) -> str:
    return f"{class_}_{cell_id}_{task_json_id}.png"


def write_thumbnail(
    class_dir: Path,
    thumbnail_file_name: str,
    image: Image.Image,
    max_num_files_per_subdir: int = 1000,
):
    """
    write the thumbnail to the class_dir, being aware of the number of thumbnails in each dir,
    and creating new subdirs if needed
    """
    dirs = [p for p in class_dir.iterdir() if p.is_dir()]

    # if there are no subdirs, create one
    if len(dirs) == 0:
        (class_dir / "0").mkdir()
        dirs.append(class_dir / "0")

    # place the thumbnail in the first subdir that has space
    for subdir in dirs:
        num_files_in_subdir = len(list(subdir.iterdir()))
        if num_files_in_subdir < max_num_files_per_subdir:
            image.save(class_dir / subdir / thumbnail_file_name)
            return

    # there were no subdirs w/ space, so create a new one
    # naive new dirname, but whatever
    new_dirname = str(len(dirs))
    (class_dir / new_dirname).mkdir()
    image.save(class_dir / new_dirname / thumbnail_file_name)


def create_thumbnails_from_tasks(
    tasks_json_path: Path,
    class_dirs: Dict[str, Path],
    task_json_id: Optional[str] = None,
    classes_to_ignore: List[str] = [],
):
    task_json_id = task_json_id or tasks_json_path.parent.name

    with open(tasks_json_path) as f:
        tasks = json.load(f)

    for task in tasks:
        image_url = task["data"]["image"]

        # task.json files hold image urls that are relative to LFM_scope
        image_path = LFM_SCOPE_PATH / image_url.replace("http://localhost:8081/", "")
        image = np.array(Image.open(image_path).convert("L"))

        img_h, img_w = image.shape

        for prediction in task["predictions"][0]["result"]:
            class_ = prediction["value"]["rectanglelabels"][0]

            if class_ in classes_to_ignore:
                continue

            cell_id = prediction["id"]
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
            write_thumbnail(
                class_dir,
                create_thumbnail_name(class_, cell_id, task_json_id),
                pil_cell_image,
            )


def create_thumbnails_from_tasks_maps(
    path_to_output_dir: Path,
    task_and_label_paths: List[Dict[str, Union[str, int]]],
    tasks_dir: Path,
    class_dirs: Dict[str, Path],
    classes_to_ignore: List[str] = [],
):
    N = int(math.log(len(task_and_label_paths), 10)) + 1
    id_to_tasks_and_labels_path: Dict[str, Dict[str, Union[str, int]]] = {}

    for tlp in tqdm(
        task_and_label_paths,
        total=len(task_and_label_paths),
        desc="creating thumbnails",
    ):
        i = tlp["task_num"]
        create_thumbnails_from_tasks(
            tasks_dir / str(tlp["task_name"]),
            class_dirs,
            task_json_id=f"{i:0{N}}",
            classes_to_ignore=classes_to_ignore,
        )
        id_to_tasks_and_labels_path[f"{i:0{N}}"] = tlp

    with open(path_to_output_dir / "id_to_task_path.json", "w") as f:
        json.dump(id_to_tasks_and_labels_path, f)
