"""
Author: IJ
Date: 2024-07-23

Description: 

This script matches the bounding box coordinates of thumbnails in completed folders of `LFM_scope/thumbnail-corrections`
with the closest match in the corresponding `labels.txt` file. It then, in a copy of the original label file, updates the label of that thumbnail to
its human-verified class. This new label file is located in `LFM_scope/merged-labels`. Additionally, another copy of the label file is stored in a folder
next to the normal `labels` folder, called `labels_plus`. This file contains all the same information as the the normal `label.txt` files, but has one
additional column which stores the status of whether a given object was machine labelled or human labelled.
"""

import csv
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import json
import os
from pathlib import Path
import re
import shutil
import traceback
from typing import cast, Tuple, List, Dict

from tqdm import tqdm

from lfm_data_utilities import YOGO_CLASS_ORDERING

IMG_W = 1032
IMG_H = 772


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


def get_all_completed_thumbnails(tld: Path):
    """
    Go through the specified thumbnail-corrections folder and get all the thumbnails
    (i.e all those within corrected_* and *-completed-* folders).

    If there are any folders within the normal "healthy/ring/etc." folders which have not been marked as complete,
    this function will raise an error and list those folders which are not yet complete.
    """

    corrections_folders = [f"{tld / ('corrected_' + c)}" for c in YOGO_CLASS_ORDERING]
    existing_correct_folders = [f"{tld / c}" for c in YOGO_CLASS_ORDERING]

    all_thumbnail_paths = []

    # Go through each of the "correction_*" folders and get all the files within them
    for folder in tqdm(corrections_folders, desc="Loading corrected_* thumbnails"):
        folder = Path(folder)
        if not folder.exists():
            print(f"Warning: folder doesn't exist: {folder}")
        else:
            # Get all the files within this folder which are not hidden files
            for file in folder.rglob("*"):
                if file.is_file() and not file.name.startswith("."):
                    all_thumbnail_paths.append(file)

    # Do the above but with the normal "healthy/ring/etc. folders."
    # Do an additional verification if these have been marked as 'complete'
    potentially_incomplete = []
    for folder in tqdm(
        existing_correct_folders, desc="Loading healthy/ring/etc. thumbnails"
    ):
        folder = Path(folder)
        if not folder.exists():
            print(f"Warning: folder doesn't exist: {folder}")
        else:
            # Get all the files within this folder which are not hidden files
            for file in folder.rglob("*"):
                if file.is_file() and not file.name.startswith("."):
                    if not "complete" in file.parent.stem:
                        potentially_incomplete.append(file)
                    else:
                        all_thumbnail_paths.append(file)

    # Display those folders which have not been marked as complete
    unique_folders = []
    for x in potentially_incomplete:
        if x.parent not in unique_folders:
            unique_folders.append(x.parent)

    # Do not continue if there are any folders which have not been marked as complete
    if len(potentially_incomplete) > 0:
        print(f"Found {len(unique_folders)} potentially incomplete folders: ")
        for folder in unique_folders:
            print(folder)
        # raise ValueError(
        #     "There appear to be some folders which have not been marked as complete. Please verify."
        # )

    # Otherwise, return all the thumbnail paths
    return all_thumbnail_paths


def group_thumbnail_paths(tp: List[Path]) -> Dict[str, List[Path]]:
    """
    Groups thumbnail paths by their task id.

    All thumbnails which share the same task id are from the same dataset.
    """

    def get_end_digit(path):
        filename = os.path.basename(path)  # Get the filename from the path
        number_part = filename.split("_")[-1].rstrip(".png")  # Extract the numeric part
        return int(number_part)  # Convert to integer for proper numeric sorting

    # Sort paths by the numeric part at the end of the filename
    sorted_paths = sorted(tp, key=get_end_digit)

    # Group paths by the numeric part at the end of the filename
    grouped_paths = {
        key: list(group) for key, group in groupby(sorted_paths, key=get_end_digit)
    }

    return grouped_paths


def get_task_from_id(tld_path: Path, task_id: int) -> Path:
    """Get the tasks file for a given task id."""
    return tld_path / "tasks" / f"thumbnail_correction_task_{task_id}.json"


def open_task(task_path: Path) -> Dict:
    """Convenience function to open a task file."""
    with open(task_path, "r") as f:
        return json.load(f)


def find_cell_indices_id_map(tasks: Dict) -> Dict[str, Dict[str, int]]:
    d = dict()
    for i, image_prediction in enumerate(tasks):
        bbox_predictions = image_prediction["predictions"][0]["result"]  # lol
        for j, bbox_prediction in enumerate(bbox_predictions):
            d[bbox_prediction["id"]] = {"image_index": i, "bbox_index": j}
    return d


def get_img_path_and_bbox_coords(task: Dict, indexes_by_id: Dict, cell_id: str):
    original_cell_prediction_indices = indexes_by_id[cell_id]
    image_index_in_task_file = original_cell_prediction_indices["image_index"]
    bbox_index_in_task_file = original_cell_prediction_indices["bbox_index"]
    image_path = Path(task[image_index_in_task_file]["data"]["image"])
    bbox_pred = task[image_index_in_task_file]["predictions"][0]["result"][
        bbox_index_in_task_file
    ]["value"]
    img_w = task[image_index_in_task_file]["predictions"][0]["result"][
        bbox_index_in_task_file
    ]["original_width"]
    img_h = task[image_index_in_task_file]["predictions"][0]["result"][
        bbox_index_in_task_file
    ]["original_height"]

    x_min, y_min = bbox_pred["x"], bbox_pred["y"]
    w, h = bbox_pred["width"], bbox_pred["height"]

    # Yolo format
    cx = x_min + w / 2
    cy = y_min + h / 2

    cx /= 100
    cy /= 100
    w /= 100
    h /= 100

    return (image_path, cx, cy, w, h)


def get_dataset_name_and_image_idx_from_image_path(img_path: Path):
    dataset_name = img_path.parent.parent.stem
    img_idx = int(img_path.stem.split("_")[-1])

    return dataset_name, img_idx


def get_yogo_label_path(model_name: str, dataset_name: str):
    p = Path(
        f"/hpc/projects/group.bioengineering/LFM_scope/yogo_labels/{model_name}/{dataset_name}"
    )

    if not p.exists():
        print(
            f"Couldn't find the labels using the default dataset name, attempting to find with regex: {dataset_name}"
        )
        ds = find_labels_using_regex(model_name, dataset_name)
        p = Path(
            f"/hpc/projects/group.bioengineering/LFM_scope/yogo_labels/{model_name}/{ds}"
        )

    return p


def get_label_paths_and_dataset_names_from_id_to_task_path_file(
    tld_path: Path,
) -> List[Path]:
    with open(tld_path / "id_to_task_path.json") as f:
        id_to_task_path = json.load(f)
        id_to_task_path = cast(Dict[str, Dict[str, str]], id_to_task_path)

    all_label_paths = []
    all_dataset_names = []
    for x in id_to_task_path:
        label_path = Path(id_to_task_path[x]["label_path"])
        ds_name = label_path.parent.stem

        all_label_paths.append(label_path)
        all_dataset_names.append(ds_name)

    return all_label_paths, all_dataset_names


def find_labels_using_regex(model_name, dataset_name):
    """
    This function only exists because for some reason, some of the older id_to_task_path.json files have label paths which have
    additional words appended to the default dataset name.
    """

    # Regex pattern to match the full date and time format YYYY-MM-DD-HHMMSS
    pattern = r"(\d{4}-\d{2}-\d{2}-\d{6})"
    dataset_date = re.search(pattern, dataset_name).group(0)

    if dataset_date is None:
        raise ValueError(f"Date not found in {dataset_name}")

    folders = [
        x
        for x in sorted(
            os.listdir(
                f"/hpc/projects/group.bioengineering/LFM_scope/yogo_labels/{model_name}"
            )
        )
        if dataset_date in x
    ]

    if len(folders) > 1:
        print(f"Multiple folders found for {dataset_date}: {folders}")
        folder = None
    elif len(folders) == 0:
        print(f"No folders found for {dataset_date}")
        folder = None
    else:
        folder = folders[0]
        print(f"Found: {folder}")
    return folder


def check_if_all_datasets_have_labels_made(dataset_names: List[Path], model_name: str):
    """
    A convenience function to go through `LFM_scope/yogo_labels` and verify that machine labels have already been
    generated for teh specified datasets.
    """

    datasets_without_labels = []
    for ds in dataset_names:
        if not (get_yogo_label_path(model_name, ds)).exists():
            datasets_without_labels.append(ds)

    if len(datasets_without_labels) > 0:
        print(
            f"The following datasets have not had YOGO labels made for them with {model_name}:"
        )
        for ds in datasets_without_labels:
            print(ds)
        raise ValueError("Not all datasets have labels made. Please verify.")


def get_verified_class_from_thumbnail_path(thumbail_path: Path):
    """
    Based on the folder path of the thumbail,
    we know what its original class was (based on its filename),
    as well as what its verified class is (based on the folder it is currently in).

    For example, if the thumbnail name is "ring_blahblah.png" and it is in the
    "corrected_healthy" folder, we know that it was machine labelled as ring
    but is actually a healthy cell based on human verification.

    Similarly, if the thumbnail's name is "ring_blahblah.png" and it remains in
    the "ring/*-completed-*" folder, then it was correctly classified as a ring
    by the model and has been human verified.
    """

    if "complete" in thumbail_path.parent.stem:
        return thumbail_path.parent.parent.stem
    elif "corrected" in thumbail_path.parent.stem:
        return thumbail_path.parent.stem
    else:
        raise ValueError(f"Thumbnail path {thumbail_path} is not in a valid folder.")


def calc_iou(x11, y11, x12, y12, x21, y21, x22, y22):
    """
    Calculate the intersection over union (IoU) of two bounding boxes,
    """

    # Get the coordinates of the intersection rectangle
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)

    # Calculate the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of both bounding boxes
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    # Calculate the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def convert_yolo_to_pixels(cx, cy, w, h):
    cx *= IMG_W
    cy *= IMG_H
    w *= IMG_W
    h *= IMG_H

    x1 = max(cx - w / 2, 0)
    y1 = max(cy - h / 2, 0)
    x2 = min(cx + w / 2, IMG_W)
    y2 = min(cy + h / 2, IMG_H)

    return x1, y1, x2, y2


def get_iou_tolerance(corrected_class):
    if corrected_class in ["wbc", "misc"]:
        return 0.6
    else:
        return 0.8


def find_best_match(label_path, corrected_class, cx, cy, w, h):
    """
    Attempts to find the closest match (based on bounding box coordinates) in the label file,
    and returns whether the match is within a certain tolerance of the thumbnail's coordinates.

    If a match is found, this function returns the line number of the match in the label file.
    """

    # Open the file
    with open(label_path, "r") as f:
        lines = f.readlines()

    # Find the closest match using intersection over union (IoU)
    max_iou = 0
    for i, line in enumerate(lines):
        class_name, x, y, width, height = line.split(" ")
        x, y, width, height = map(float, [x, y, width, height])
        iou = calc_iou(
            *convert_yolo_to_pixels(cx, cy, w, h),
            *convert_yolo_to_pixels(x, y, width, height),
        )

        if iou >= max_iou:
            max_iou = iou
            line_idx = i

    # Check if the closest match is within the desired IoU tolerance
    iou_tol = get_iou_tolerance(corrected_class)

    good_match = False
    if max_iou >= iou_tol:
        good_match = True
    else:
        line_idx = None

    return good_match, line_idx, max_iou


def copy_file(src_file, dest_dir):
    shutil.copy(src_file, dest_dir)


def copy_files_concurrently(src_dir, dest_dir, orig_label_path, num_workers=10):
    # Ensure the destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the source directory
    files = [file for file in Path(src_dir).iterdir() if file.is_file()]

    # List all files in the original labels directory
    orig_files = [
        file.stem for file in Path(orig_label_path).iterdir() if file.is_file()
    ]

    # Only copy the files which are in the original labels directory
    files = [file for file in files if file.stem in orig_files]

    # Use ThreadPoolExecutor to copy files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(copy_file, files, [dest_dir] * len(files)),
                total=len(files),
                desc=f"\nCopying files from {str(src_dir).replace('/hpc/projects/group.bioengineering/', '')} to {str(dest_dir).replace('/hpc/projects/group.bioengineering/', '')}",
            )
        )


def add_zero_to_label_files(directory: Path):
    for file_path in tqdm(
        directory.glob("*.txt"), desc=f"Adding zeros label files in {directory.stem}"
    ):
        lines = file_path.read_text().splitlines()

        # Need to handle the exception case where a dataset shows up twice and so we want to avoid
        # adding the extra column more than once

        if len(lines[0].split(" ")) == 6:
            continue

        with file_path.open("w") as file:
            for line in lines:
                file.write(f"{line} 0\n")


def update_labels_files(
    new_class: str, label_path: Path, label_plus_path: Path, line_idx: int
):
    """Given a specific label (a row index within a labels.txt file), update the class ID to the new class ID."""

    # Get the class ID
    class_name = new_class if "corrected" not in new_class else new_class.split("_")[1]

    class_id = YOGO_CLASS_ORDERING.index(class_name)

    # Open the file
    with open(label_path, "r") as f:
        lines = f.readlines()
    line_to_update = lines[line_idx]
    line_vals = line_to_update.split(" ")
    orig_class_name, x, y, width, height = line_vals
    new_line = " ".join([str(class_id), x, y, width, height])

    lines[line_idx] = new_line

    # Save the updated label file
    with open(label_path, "w") as f:
        for x in lines:
            f.write(x)

    # Open the file
    with open(label_plus_path, "r") as f:
        lines = f.readlines()

    line_to_update = lines[line_idx]
    line_vals = line_to_update.split(" ")
    orig_class_name, x, y, width, height, machine_or_human = line_vals
    new_line_labels_plus = (
        " ".join([str(class_id), x, y, width, height.strip(), str(1)]) + "\n"
    )
    lines[line_idx] = new_line_labels_plus

    # Save the updated label_plus file
    with open(label_plus_path, "w") as f:
        for x in lines:
            f.write(x)


if __name__ == "__main__":
    # Argparse a folder and model name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "thumbnail_folder", type=str, help="The folder containing the thumbnails"
    )
    parser.add_argument("model_name", type=str, help="The name of the model")
    args = parser.parse_args()

    # First check that all the datasets in this folder have labels made
    # This will raise an error if labels are missing
    (
        label_paths,
        dataset_names,
    ) = get_label_paths_and_dataset_names_from_id_to_task_path_file(
        Path(args.thumbnail_folder)
    )
    check_if_all_datasets_have_labels_made(dataset_names, args.model_name)

    # Get all thumbnails in this folder
    # This will raise an error if there are incomplete folders
    all_thumbnail_paths = get_all_completed_thumbnails(Path(args.thumbnail_folder))
    grouped_thumbnail_paths = group_thumbnail_paths(all_thumbnail_paths)

    # Let's create the folder in merged_labels for this model and dataset if it doesn't exist
    all_merged_label_paths = []
    all_merged_label_plus_paths = []

    print(f"Copying label files to merged_labels...")
    for i, (lp, ds) in enumerate(zip(label_paths, dataset_names)):
        # Create a metadata folder to house information on the merged labels (IoU and etc.)
        metadata_path = Path(
            f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/metadata_on_merged_labels"
        )
        metadata_path.mkdir(parents=True, exist_ok=True)

        # Copy the labels from yogo_labels to merged_labels
        yogo_label_path = get_yogo_label_path(args.model_name, ds)

        cleaned_ds_name = yogo_label_path.stem

        merged_label_path = Path(
            f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/{cleaned_ds_name}/labels"
        )
        merged_label_path.mkdir(parents=True, exist_ok=True)

        merged_label_plus_path = Path(
            f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/{cleaned_ds_name}/labels_plus"
        )
        merged_label_plus_path.mkdir(parents=True, exist_ok=True)

        all_merged_label_paths.append(merged_label_path)
        all_merged_label_plus_paths.append(merged_label_plus_path)

        # Copy all files from the source directory to the destination directory using the concurrent function
        # Note we also pass in the original label (from the id_to_task_path.json) so that only those labels are copied and not any extra ones
        # This is done because when machine labels are generated (those stored in yogo_labels), we pass it an image path to yogo infer
        # and the numbers of images previously generated may be greater than the number of labels' files generated.
        copy_files_concurrently(Path(yogo_label_path), merged_label_path, lp)

        # Copy all files again to the 'labels_plus' folder
        copy_files_concurrently(Path(yogo_label_path), merged_label_plus_path, lp)

        # Update the label files with an additional column, defaulted to 0s, indicating that these are machine labels
        add_zero_to_label_files(merged_label_plus_path)

        print(f"\nDone copying {i+1} / {len(label_paths)}")

    try:
        for task_id in tqdm(
            grouped_thumbnail_paths.keys(), desc="Looping through task ids..."
        ):
            task_file = open_task(
                get_task_from_id(Path(args.thumbnail_folder), task_id)
            )
            indexes_by_id = find_cell_indices_id_map(task_file)

            # Keep a tally
            class_to_total = {c: 0 for c in YOGO_CLASS_ORDERING}
            class_to_matches = {c: 0 for c in YOGO_CLASS_ORDERING}

            # Create a CSV file for each task_id
            thumbnail = grouped_thumbnail_paths[task_id][0]
            class_name, cell_id, _ = parse_thumbnail_name(thumbnail.name)
            img_path, cx, cy, w, h = get_img_path_and_bbox_coords(
                task_file, indexes_by_id, cell_id
            )
            dataset_name, _ = get_dataset_name_and_image_idx_from_image_path(img_path)

            csv_file = open(metadata_path / f"{dataset_name}.csv", "w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "base_label_model",
                    "dataset_name",
                    "img_idx",
                    "orig_class",
                    "corrected_class",
                    "best_iou",
                    "iou_tolerance_perc",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "img_path",
                ]
            )

            for thumbnail in tqdm(
                grouped_thumbnail_paths[task_id], desc="Looping through thumbnails..."
            ):
                class_name, cell_id, _ = parse_thumbnail_name(thumbnail.name)
                img_path, cx, cy, w, h = get_img_path_and_bbox_coords(
                    task_file, indexes_by_id, cell_id
                )
                dataset_name, img_idx = get_dataset_name_and_image_idx_from_image_path(
                    img_path
                )
                yogo_label_path = get_yogo_label_path(args.model_name, dataset_name)
                corrected_class = get_verified_class_from_thumbnail_path(thumbnail)

                cleaned_ds_name = yogo_label_path.stem
                merged_label_path = Path(
                    f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/{cleaned_ds_name}/labels/img_{img_idx:05d}.txt"
                )
                merged_label_plus_path = Path(
                    f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/{cleaned_ds_name}/labels_plus/img_{img_idx:05d}.txt"
                )

                # For some reason we used to label files with only four 0s of padding...
                if not merged_label_path.exists():
                    merged_label_path = Path(
                        f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/{cleaned_ds_name}/labels/img_{img_idx:04d}.txt"
                    )
                    merged_label_plus_path = Path(
                        f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/{cleaned_ds_name}/labels_plus/img_{img_idx:04d}.txt"
                    )

                # Find the best match in the label file
                match_found, closest_match_line_idx, iou = find_best_match(
                    merged_label_path,
                    corrected_class,
                    cx,
                    cy,
                    w,
                    h,
                )

                # If a match is found, update the corresponding files in both the labels and labels_plus
                if match_found:
                    update_labels_files(
                        corrected_class,
                        merged_label_path,
                        merged_label_plus_path,
                        closest_match_line_idx,
                    )

                # Log the thumbnail in the metadata file
                class_name = (
                    corrected_class
                    if "corrected" not in corrected_class
                    else corrected_class.split("_")[1]
                )
                x1, y1, x2, y2 = convert_yolo_to_pixels(cx, cy, w, h)
                writer.writerow(
                    [
                        args.model_name,
                        dataset_name,
                        img_idx,
                        class_name,
                        corrected_class,
                        iou,
                        get_iou_tolerance(corrected_class),
                        x1,
                        y1,
                        x2,
                        y2,
                        str(img_path).replace(
                            "http:/localhost:8081/",
                            "/hpc/projects/group.bioengineering/LFM_scope/",
                        ),
                    ]
                )
                class_to_total[class_name] += 1

                if match_found:
                    class_to_matches[class_name] += 1
            csv_file.close()

            print(f"{'='*10}")
            print("Class: matches / total (%)")
            print(f"{'='*10}")
            for k in class_to_matches.keys():
                matches = class_to_matches[k]
                total = class_to_total[k]
                print(f"{k}: {matches} / {total} = {matches/total:.2f}")
            print(f"\n{class_to_matches=}")
            print(f"{class_to_total=}")

        print("X-Y matching complete.")

        # Now, let's go through and check for any labels files which have no human verified labels
        # and remove those (since they consist entirely of machine labels)
        print(
            "Checking for and removing any label files which do not have any human verified labels now..."
        )

        for lbl_path, lbl_plus_path in zip(
            all_merged_label_paths, all_merged_label_plus_paths
        ):
            # Go through each file in the lbl_plus_path folder and check if all of its last columns are 0
            # if so, then there are no human verified labels and that file (and its corresponding file in lbl_path)
            # should be removed
            counter = 0
            for file in lbl_plus_path.glob("*.txt"):
                lines = file.read_text().splitlines()
                if all([line.split(" ")[-1] == "0" for line in lines]):
                    counter += 1
                    os.remove(lbl_path / file.name)
                    os.remove(file)
            if counter > 1:
                print(
                    f"{counter} files were removed from {lbl_path} for not having any human verified labels."
                )
            elif counter == 1:
                print(
                    f"{counter} file was removed from {lbl_path} for not having any human verified labels."
                )

    except Exception as e:
        print(traceback.format_exc())

        print(f"\nUnexpected error occurred: {e}")
        print("Removing all merged labels...")
        for lp1 in all_merged_label_paths:
            shutil.rmtree(lp1.parent)
            dataset = lp1.parent.stem
            os.remove(
                f"/hpc/projects/group.bioengineering/LFM_scope/merged_labels/{args.model_name}/metadata_on_merged_labels/{dataset}.csv"
            )
