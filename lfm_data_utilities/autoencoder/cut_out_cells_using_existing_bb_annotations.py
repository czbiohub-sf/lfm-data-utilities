"""
Using existing bounding box annotations (stored in label files),
crop out cells and put them into a folder.
"""

from typing import List, Optional
from pathlib import Path

import numpy as np
import cv2

from lfm_data_utilities.utils import (
    get_all_dataset_paths,
    Dataset,
    load_img,
    load_label_file,
    Segment,
    Point,
    ImageAndLabelPathPair,
)

# healthy / ring / troph / schizont / gametocyte / wbc
ALLOWABLE_LABELS = [0, 1, 2, 3, 4, 5]


def get_corresponding_label_dir(
    img_dir: Path, search_dir: Path
) -> Optional[ImageAndLabelPathPair]:
    dataset_dir = img_dir.parent.name
    try:
        lbl_dir_parent = next(search_dir.rglob(f"{dataset_dir}*"))
        return ImageAndLabelPathPair(img_dir, Path(lbl_dir_parent / "labels"))
    except:
        print(f"Label dir not found in {search_dir} for directory: {img_dir.parent}")
        print("Continuing...")
        return None


def get_cell_thumbnails_from_dataset(
    img_label_pair: ImageAndLabelPathPair, save_loc: Path
) -> None:
    if img_label_pair is None:
        return

    img_path = img_label_pair.img_path
    lbl_path = img_label_pair.lbl_path
    img = load_img(img_path)
    w, h = img.shape
    segments = load_label_file(lbl_path, h, w)
    slices = get_img_slices(img, segments)
    save_slices(segments, slices, img_path.parent, save_loc)


def save_slices(
    segments: List[Segment],
    slices: List[np.ndarray],
    dataset_path: Path,
    save_loc: Path,
) -> None:
    for segment, slice in zip(segments, slices):
        cell_type = int(segment.classification)
        tl, br = segment.top_left, segment.bottom_right
        fc = segment.frame_count

        if cell_type in ALLOWABLE_LABELS and all(slice.shape):
            save_slice(slice, cell_type, tl, br, fc, dataset_path, save_loc)


def save_slice(
    slice: np.ndarray,
    type: int,
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
    img_paths = list(Path(args.path_to_experiments).rglob("images/"))
    search_dir = Path(args.label_path)
    save_loc = args.save_loc_path

    img_label_path_pairs = [
        get_corresponding_label_dir(x, search_dir) for x in img_paths
    ]

    if not Path(save_loc).exists():
        Path.mkdir(save_loc)

    for pair in img_label_path_pairs:
        get_cell_thumbnails_from_dataset(pair, save_loc)
