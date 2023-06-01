"""
Using existing bounding box annotations (stored in label files),
crop out cells and put them into a folder.
"""

from functools import partial
from multiprocessing import Pool
import os
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import json

import numpy as np
import cv2
from tqdm import tqdm

from lfm_data_utilities.utils import (
    load_img,
    load_label_file,
    Segment,
    Point,
    get_img_and_label_pairs,
)

DEFAULT_LABELS_SEARCH_DIR = (
    "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels"
)


def get_corresponding_dataset_dir_in_search_dir(
    dataset_dir: Path, search_dir: Path
) -> Optional[Path]:
    dataset_dir_search_string = dataset_dir.name

    # If an images folder is passed in directly
    if search_dir.stem == "labels" and search_dir.is_dir():
        if dataset_dir_search_string in search_dir.parent.name:
            return search_dir.parent
        else:
            print(
                f"WARNING: {search_dir} was passed in which doesn't seem to match {dataset_dir}. Making thumbnails anyway but this seems weird..."
            )
            return search_dir.parent

    # If a single experiment folder was passed
    if "labels" in os.listdir(search_dir):
        if (Path(search_dir) / "labels").is_dir():
            if dataset_dir_search_string in search_dir.name:
                return search_dir
        else:
            print(
                f"WARNING: {search_dir} was passed in which doesn't seem to match {dataset_dir}. Making thumbnails anyway but this seems weird..."
            )
            return search_dir

    # Recursively search multiple directories for the folder with the corresponding labels folder
    try:
        ds_dir_in_search = next(search_dir.rglob(f"{dataset_dir_search_string}*"))
        return Path(ds_dir_in_search)
    except StopIteration:
        print(f"Couldn't find {dataset_dir_search_string} in {search_dir.stem}")
        return None


def get_class_map(dataset_dir: Path) -> Dict[int, str]:
    """Each dataset might have its own order for classes. This function returns the mapping
    from this dataset's order to the 'general' order.

    Returns
    -------
    Dict [int, str]:
        Integer id to string (i.e healthy / ring / troph / etc.)
    """

    classes_file = dataset_dir / "notes.json"
    if not classes_file.exists():
        raise ValueError(f"No classes file exists for this directory: {dataset_dir}")

    try:
        with open(classes_file.absolute(), "r") as f:
            vals = json.load(f)
            d = {int(x["id"]): x["name"] for x in vals["categories"]}
        return d
    except Exception:
        raise


def save_thumbnails_from_dataset(
    dataset_path: Path, label_search_dir: Path, save_loc: Path, quiet: bool = True
) -> None:
    # Get images folder path
    imgs_folder_path = dataset_path / "images"

    if not imgs_folder_path.exists():
        print(f"{imgs_folder_path} does not exist.")
        return

    # Attempt to find the corresponding dataset in the search directory for labels
    corresponding_dataset_in_labels_dir = get_corresponding_dataset_dir_in_search_dir(
        dataset_path, label_search_dir
    )

    # Verify that the labels folder exists in the search directory, otherwise return
    if corresponding_dataset_in_labels_dir is not None:
        labels_path: Path = (
            get_corresponding_dataset_dir_in_search_dir(dataset_path, label_search_dir)
            / "labels"
        )
        if not labels_path.exists():
            print(f"{labels_path} does not exist.")
            return
        else:
            # Inform user
            print(f"Working on: {dataset_path.parent.name}/{dataset_path.name}")
            print(
                f"Corresponding labels folder found in: {corresponding_dataset_in_labels_dir}"
            )
    else:
        if not quiet:
            print("No corresponding labels folder found in the search directory.")
        return

    # Get class mapping for this dataset
    try:
        class_map = get_class_map(corresponding_dataset_in_labels_dir)
        for c in class_map.values():
            if not Path(save_loc / c).exists():
                Path.mkdir(Path(save_loc / c))

    except Exception as e:
        print(f"Error reading classes.txt for {dataset_path}. Error: {e}")
        return

    image_label_pairs = get_img_and_label_pairs(imgs_folder_path, labels_path)

    for img_path, lbl_path in tqdm(image_label_pairs):
        try:
            img = load_img(img_path)
            h, w = img.shape
            segments = load_label_file(lbl_path, h, w)
            crops = get_img_crops(img, segments)  # Cropped regions for each cell
            save_crops(segments, crops, dataset_path, class_map, save_loc)
        except:
            print(f"Errored on {img_path, lbl_path}")
            quit()


def save_crops(
    segments: List[Segment],
    crops: List[np.ndarray],
    dataset_path: Path,
    class_map: Dict[str, str],
    save_loc: Path,
) -> None:
    segs_and_crops: List[Tuple[Segment, np.ndarray]] = zip(segments, crops)

    for segment, crop in segs_and_crops:
        try:
            cell_type = class_map[int(segment.classification)]
        except Exception as e:
            print("Error when performing class map.")
            print(f"Dataset: {dataset_path}")
            print(f"Frame: {segment.frame_count}")
            print(f"Class map: {class_map}")
            print(f"Received classification integer: {int(segment.classification)}")
            print(e)
            raise
        tl, br = segment.top_left, segment.bottom_right
        fc = segment.frame_count

        if all(crop.shape):
            save_crop(crop, cell_type, tl, br, fc, dataset_path, save_loc)


def save_crop(
    crop: np.ndarray,
    type: str,
    tl: Point,
    br: Point,
    frame_count: int,
    dataset_path: Path,
    save_loc: Path,
) -> None:
    """Save the crop and store relevant metadata in the filename.

    The filename of the thumbnail is in the following format. Metadata is delinated by underscores.

        {cell type} _
        {original dataset name} _
        {top left bounding box x coordinate} _
        {frame count} _
        {top left bounding box y coordinate) _
        {bottom right bb x coordinate} _
        {bottom right bb y coordinate}

    Parameters
    ----------
    crop: np.ndarray
        crop of the original image to be saved
    type: int
        Classification (i.e healthy, ring, etc.)
    tl: Point
        Top left of the bounding box
    br: Point
        Bottom right of the bounding box
    frame_count: int
        Which frame this image / set of labels is from
    dataset_path: Path
    save_loc: Path
    """

    ds_name = dataset_path.name
    crop_filename = Path(
        f"{type}_{ds_name}_{frame_count}_{tl.x}_{tl.y}_{br.x}_{br.y}.png"
    )
    cv2.imwrite(str(save_loc / type / crop_filename), crop)


def get_img_crops(img: np.ndarray, segments: List[Segment]) -> List[np.ndarray]:
    """Given an image and segments (i.e bounding boxes), return crops of the image in those locations.

    Parameters
    ----------
    img: np.ndarray
    segments: List[Segment]
        A list of labels for the image (i.e containing labels and bounding box information)

    Returns
    -------
    List[np.ndarray]
        A list of crops of the image
    """
    crops = []
    for s in segments:
        tl_x, tl_y = s.top_left.x, s.top_left.y
        br_x, br_y = s.bottom_right.x, s.bottom_right.y

        crops.append(img[tl_y:br_y, tl_x:br_x])

    return crops


def get_img_paths(main_dir: Path) -> List[Path]:
    img_paths: List[Path] = []

    # Account for the case where a specific images folder is passed in
    if main_dir.name == "images":
        img_paths.append(main_dir)
        return img_paths

    # Otherwise walk through the subdirectories and find folders containing images/ folders
    for main_path, directories, _ in os.walk(main_dir):
        for d in directories:
            if "images" == d:
                img_paths.append(Path(os.path.join(main_path, d)))

    return img_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Crop out the RBCs from all the frames of experiments which already have a labels folder."
    )
    parser.add_argument(
        "path_to_experiments",
        type=Path,
        help=(
            "Provide the top-level directory (a folder containing oracle runs).\n ",
            "Can be a single folder containing an images/ folder or a high-level folder\n "
            "containing multiple folders which themselves contain images/ directories (or whose\n"
            "subfolders contain images/ directories).",
        ),
    )

    parser.add_argument(
        "--label_path",
        type=Path,
        help="Top level dir for where to search for labels",
        default=DEFAULT_LABELS_SEARCH_DIR,
        required=False,
    )

    parser.add_argument(
        "save_loc_path",
        type=Path,
        help="Path to save thumbnails. Folder will be made if it doesn't exist already",
    )

    args = parser.parse_args()
    experiment_dir = Path(args.path_to_experiments)
    search_dir = Path(args.label_path)
    save_loc = Path(args.save_loc_path)

    if not save_loc.exists():
        Path.mkdir(save_loc)

    print(f"{'='*10}")
    print(f"Searching for experiments in: {args.path_to_experiments}")
    print(f"Searching for labels in: {search_dir}")
    print(f"{'='*10}")

    img_paths = [x.parent for x in get_img_paths(experiment_dir)]

    f = partial(
        save_thumbnails_from_dataset, label_search_dir=search_dir, save_loc=save_loc
    )

    with Pool() as p:
        list(tqdm(p.imap(f, img_paths), total=len(img_paths)))

    print(f"Done! Take a look at: {save_loc} to view the thumbnails.")
