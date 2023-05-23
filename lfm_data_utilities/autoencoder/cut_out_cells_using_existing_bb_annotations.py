"""
Using existing bounding box annotations (stored in label files),
crop out cells and put them into a folder.
"""

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2

from lfm_data_utilities.utils import (
    load_img,
    load_label_file,
    Segment,
    Point,
    ImageAndLabelPathPair,
    get_img_and_label_pairs,
)


def get_corresponding_dataset_dir_in_search_dir(
    dataset_dir: Path, search_dir: Path
) -> Optional[Path]:
    dataset_dir_search_string = dataset_dir.name
    try:
        ds_dir_in_search = next(search_dir.rglob(f"{dataset_dir_search_string}*"))
        return Path(ds_dir_in_search)
    except StopIteration:
        print(f"Could not find {dataset_dir_search_string} in {search_dir}")
        return None


def get_class_map(dataset_dir: Path) -> dict:
    """Each dataset might have its own order for classes. This function returns the mapping
    from this dataset's order to the 'general' order.

    Returns
    -------
    dict [int, str]:
        Integer id to string (i.e healthy / ring / troph / etc.)
    """

    classes_file = dataset_dir / "classes.txt"
    if not classes_file.exists():
        raise ValueError(f"No classes file exists for this directory: {dataset_dir}")

    try:
        with open(classes_file.absolute(), "r") as f:
            vals = f.readlines()
            d = {i: x for i, x in enumerate([y.strip() for y in vals])}
        return d
    except Exception as e:
        raise


def save_thumbnails_from_dataset(
    dataset_path: Path, label_search_dir: Path, save_loc: Path
) -> None:
    # Get images folder path
    imgs_folder_path = dataset_path / "images"
    if not imgs_folder_path.exists():
        print(f"{imgs_folder_path} does not exist.")
        return

    # Attempt to find the corresponding dataset in the search directory for labels
    corresponding_dataset_in_labels_dir = get_corresponding_dataset_dir_in_search_dir(
        dataset_path, search_dir
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
        print("No corresponding labels folder found in the search directory.")
        return

    # Get class mapping for this dataset
    try:
        class_map = get_class_map(dataset_path)
    except Exception as e:
        print(f"Error reading classes.txt for {dataset_path}. Error: {e}")
        return

    image_label_pairs = get_img_and_label_pairs(imgs_folder_path, labels_path)

    for img_path, lbl_path in image_label_pairs:
        img = load_img(img_path)
        h, w = img.shape
        segments = load_label_file(lbl_path, h, w)
        slices = get_img_slices(img, segments)  # Cropped regions for each cell
        save_slices(segments, slices, dataset_path, class_map, save_loc)


def save_slices(
    segments: List[Segment],
    slices: List[np.ndarray],
    dataset_path: Path,
    class_map: dict[str, str],
    save_loc: Path,
) -> None:
    for segment, slice in zip(segments, slices):
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

        if all(slice.shape):
            save_slice(slice, cell_type, tl, br, fc, dataset_path, save_loc)


def save_slice(
    slice: np.ndarray,
    type: str,
    tl: Point,
    br: Point,
    frame_count: int,
    dataset_path: Path,
    save_loc: Path,
) -> None:
    """Save the slice and store relevant metadata in the filename.

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
    slice: np.ndarray
        Slice of the original image to be saved
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
    slice_filename = Path(
        f"{type}_{ds_name}_{frame_count}_{tl.x}_{tl.y}_{br.x}_{br.y}.png"
    )
    cv2.imwrite(str(save_loc / slice_filename), slice)


def get_img_slices(img: np.ndarray, segments: List[Segment]) -> List[np.ndarray]:
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
    slices = []
    for s in segments:
        tl_x, tl_y = s.top_left.x, s.top_left.y
        br_x, br_y = s.bottom_right.x, s.bottom_right.y

        slices.append(img[tl_y:br_y, tl_x:br_x])

    return slices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Crop out the RBCs from all the frames of experiments which already have a labels folder."
    )
    parser.add_argument(
        "path_to_experiments",
        type=Path,
        help="Provide the top-level directory (a folder containing oracle runs)",
    )

    parser.add_argument(
        "label_path", type=Path, help="Top level dir for where to search for labels"
    )

    parser.add_argument(
        "save_loc_path",
        type=Path,
        help="Path to save thumbnails. Folder will be made if it doesn't exist already",
    )

    args = parser.parse_args()
    search_dir = Path(args.label_path)
    save_loc = args.save_loc_path

    if not Path(save_loc).exists():
        Path.mkdir(save_loc)

    img_gen = Path(args.path_to_experiments).rglob("images/")
    for img_path in img_gen:
        dp = img_path.parent
        save_thumbnails_from_dataset(dp, search_dir, save_loc)
