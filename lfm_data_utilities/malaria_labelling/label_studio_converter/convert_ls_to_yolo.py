import re
import json
import shutil

from typing import List
from pathlib import Path

from lfm_data_utilities.utils import PathLike
from lfm_data_utilities.malaria_labelling.labelling_constants import IMG_SERVER_ROOT


def create_classes_txt(classes, output_path):
    with open(output_path, "w") as f:
        f.write("\n".join(classes))


def create_notes_json(classes, output_path):
    json_data = {
        "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
        # not sure if this is required for label studio, but I dont want to risk it
        "info": {
            "year": 2023,
            "version": "1.0",
            "contributor": "Label Studio",
        },
    }
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)


def prep_yolo_dir(
    path_to_output_dir: PathLike,
    overwrite_existing_labels: bool,
    label_dir_name: str = "labels",
):
    # necessary components for the output directory
    output_dir = Path(path_to_output_dir)
    labels_dir = output_dir / label_dir_name
    images_dir = output_dir / "images"
    notes_file = output_dir / "notes.json"
    classes_txt = output_dir / "classes.txt"

    subfiles_exist = any(
        p.exists() for p in (labels_dir, images_dir, notes_file, classes_txt)
    )

    # Check if the output directory exists, and if it's empty
    if subfiles_exist:
        if not overwrite_existing_labels:
            raise Exception(
                f"The output directory ({output_dir}) already exists. If you want to overwrite the "
                "existing labels, set overwrite_existing_labels=True."
            )
        else:
            try:
                shutil.rmtree(labels_dir)
            except FileNotFoundError:
                print(
                    f"labels dir not found for {output_dir.name}; therefore we will not delete it"
                )
            try:
                shutil.rmtree(images_dir)
            except FileNotFoundError:
                print(
                    f"images dir not found for {output_dir.name}; therefore we will not delete it"
                )
            try:
                notes_file.unlink()
            except FileNotFoundError:
                print(
                    f"notes file not found for {output_dir.name}; therefore we will not delete it"
                )
            try:
                classes_txt.unlink()
            except FileNotFoundError:
                print(
                    f"classes file not found for {output_dir.name}; therefore we will not delete it"
                )

    # Create the output directories
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, labels_dir, images_dir, notes_file, classes_txt


def convert_ls_to_yolo(
    path_to_ls_file: PathLike,
    path_to_output_dir: PathLike,  # most likely parent of path_to_ls_file
    classes: List[str],
    label_dir_name: str = "labels",
    overwrite_existing_labels: bool = False,
    download_images: bool = False,
):
    with open(path_to_ls_file, "r") as f:
        ls_data = json.load(f)

    output_dir, labels_dir, images_dir, notes_file, classes_txt = prep_yolo_dir(
        path_to_output_dir, overwrite_existing_labels, label_dir_name
    )

    create_classes_txt(classes, classes_txt)
    create_notes_json(classes, notes_file)

    for task in ls_data:
        image_url = task["data"]["image"]

        image_path = IMG_SERVER_ROOT / re.sub("http://localhost:\d+/", "", image_url)

        if download_images:
            shutil.copy(image_path, images_dir)

        image_name = image_path.name
        label_file_name = Path(image_name).with_suffix(".txt")

        with open(labels_dir / label_file_name, "w") as f:
            for bbox_prediction in task["predictions"][0]["result"]:
                x_min = bbox_prediction["value"]["x"]
                y_min = bbox_prediction["value"]["y"]
                width = bbox_prediction["value"]["width"]
                height = bbox_prediction["value"]["height"]
                class_id = classes.index(bbox_prediction["value"]["rectanglelabels"][0])

                # convert to yolo format
                x_center = x_min + width / 2
                y_center = y_min + height / 2

                x_center /= 100
                y_center /= 100
                width /= 100
                height /= 100

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
