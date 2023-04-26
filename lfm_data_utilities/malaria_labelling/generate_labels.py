#! /usr/bin/env python3

import cv2
import time
import torch


import numpy as np

from tqdm import tqdm
from pathlib import Path
from cellpose import models
from cellpose.utils import (
    fill_holes_and_remove_small_masks,
    outlines_list,
)
from typing import Optional, Literal, List, Tuple

from labelling_constants import CLASSES
from generate_dataset_def import gen_dataset_def
from generate_labelstudio_tasks import generate_tasks_for_runset
from utils import convert_coords

from yogo.infer import predict
from yogo.utils import iter_in_chunks


def empty_dir(path: Path):
    """
    recursively removes all files and directories in path, assumed to be a dir
    """
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            empty_dir(child)
            child.rmdir()


def get_outlines(
    path_to_folder: Path, chunksize: int = 32
) -> List[Tuple[Path, List[np.ndarray]]]:
    """Return a list of tuples (path to image, detection outlines)

    This should be run on the GPU, else it is painfully slow! Allocate some CPU too,
    we are doing a good amount of image processing.
    """
    model = models.Cellpose(gpu=True, model_type="cyto2", device=torch.device("cuda"))

    outlines: List[Tuple[Path, List[np.ndarray]]] = []

    image_filenames = list(path_to_folder.glob("*.png"))
    filename_iterator = iter_in_chunks(image_filenames, chunksize)

    for img_filename_chunk in tqdm(
        filename_iterator, total=len(image_filenames) // chunksize + 1
    ):
        imgs = []
        for img_path in img_filename_chunk:
            # for some pathological reason, if imread fails it returns None
            potential_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if potential_image is not None:
                imgs.append(potential_image)
            else:
                # TODO is this the right thing? ignore weird files? I think so...
                print(
                    f"File {img_path} cannot be interpreted as an image (cv2.imread failed)"
                )

        per_img_masks, _flows, _styles, _diams = model.eval(imgs, channels=[0, 0])

        for file_path, masks in zip(img_filename_chunk, per_img_masks):
            masks = fill_holes_and_remove_small_masks(masks)
            mask_outlines = outlines_list(masks)
            outlines.append((file_path, mask_outlines))

    return outlines


def outlines_to_yogo_labels(label_dir_path, outlines, label):
    for file_path, image_outlines in outlines:
        label_file_name = str(label_dir_path / file_path.with_suffix(".txt").name)

        with open(label_file_name, "w") as f:
            for outline in image_outlines:
                xmin, xmax, ymin, ymax = (
                    outline[:, 0].min(),
                    outline[:, 0].max(),
                    outline[:, 1].min(),
                    outline[:, 1].max(),
                )
                try:
                    xcenter, ycenter, width, height = convert_coords(
                        xmin, xmax, ymin, ymax
                    )
                except ValueError as e:
                    # xmin == xmax or ymin == ymax, so just ignore that label
                    print(f"exception {e} found; ignoring label")
                    continue
                f.write(f"{label} {xcenter} {ycenter} {width} {height}\n")


def label_folder_with_cellpose(
    path_to_images: Path, label_dir_name="labels", chunksize=32, label=0
):
    # Write classes.txt for label studio
    with open(str(path_to_images.parent / "classes.txt"), "w") as f:
        for clss in CLASSES:
            f.write(f"{clss}\n")

    path_to_label_dir = path_to_images.parent / label_dir_name
    path_to_label_dir.mkdir(exist_ok=True, parents=True)

    outlines = get_outlines(path_to_images, chunksize=chunksize)

    outlines_to_yogo_labels(path_to_label_dir, outlines, label)


def label_folder_with_yogo(
    path_to_images: Path,
    path_to_pth: Path,
    label_dir_name="labels",
    chunksize=32,
    label=0,
    **kwargs,
):
    # Write classes.txt for label studio
    with open(str(path_to_images.parent / "classes.txt"), "w") as f:
        for clss in CLASSES:
            f.write(f"{clss}\n")

    path_to_label_dir = path_to_images.parent / label_dir_name
    path_to_label_dir.mkdir(exist_ok=True, parents=True)

    predict(
        path_to_pth,
        path_to_images=path_to_images,
        output_dir=path_to_label_dir,
        thresh=0.5,
    )


def label_runset(
    path_to_runset_folder: Path,
    model: Literal["cellpose", "yogo"] = "cellpose",
    path_to_pth: Optional[Path] = None,
    label_dir_name: str = "labels",
    chunksize=32,
    label=0,
    skip=True,
) -> List[Path]:
    print(
        f"starting to label run set; {'skipping' if skip else 'overwritting'} existing labels"
    )
    print("finding directories to label...")
    paths_to_images = list(path_to_runset_folder.glob("./**/images"))
    print(f"found {len(paths_to_images)} directories to label")

    good_run_folders = []
    for i, path_to_images in enumerate(paths_to_images, start=1):
        print(
            f"{i} / {len(paths_to_images)} | {path_to_images.parent.name}", end="    "
        )
        t0 = time.perf_counter()

        label_dir = path_to_images.parent / label_dir_name
        if label_dir.exists() and len(list(label_dir.iterdir())) > 0:
            if skip:
                print(f"skipping label directory {label_dir.name}...")
                continue

            # we are overwriting the labels
            print(f"overwriting label directory {label_dir.name}...")
            empty_dir(label_dir)

        good_run_folders.append(path_to_images.parent)

        if model == "cellpose":
            label_fn = label_folder_with_cellpose
            args = (path_to_images,)
            kwargs = {
                "chunksize": chunksize,
                "label": label,
                "label_dir_name": label_dir_name,
            }
        elif model == "yogo":
            label_fn = label_folder_with_yogo
            args = (
                path_to_images,
                path_to_pth,
            )
            kwargs = {
                "chunksize": chunksize,
                "label": label,
                "label_dir_name": label_dir_name,
            }
        else:
            raise ValueError(
                f"only valid options for model is 'cellpose' or 'yogo': got {model}"
            )

        try:
            label_fn(*args, **kwargs)
        except Exception:
            import traceback

            traceback.print_exc()

        print(f"{time.perf_counter() - t0:.0f}s")

    return good_run_folders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("label a set of run folders")
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument(
        "--existing-label-action",
        "-e",
        choices=["skip", "overwrite"],
        default="skip",
        help="skip or overwrite existing labels when encountered, defaults to 'skip'",
    )
    parser.add_argument(
        "--model",
        choices=["cellpose", "yogo"],
        default="cellpose",
        help="choose cellpose or yogo to label the data",
    )
    parser.add_argument(
        "--path-to-yogo-pth",
        default=None,
        help="path to pth file for the yogo model; required if yogo is the selected model",
    )
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
    if args.model == "yogo" and args.path_to_yogo_pth is None:
        raise ValueError(
            "path to pth file for the yogo model is required if "
            "yogo is the selected model - see --path-to-yogo-pth option"
        )

    print("labelling runset...")
    t0 = time.perf_counter()
    labelled_run_paths = label_runset(
        path_to_runset,
        model=args.model,
        path_to_pth=args.path_to_yogo_pth,
        label=CLASSES.index("healthy"),
        label_dir_name=args.label_dir_name,
        skip=args.existing_label_action == "skip",
    )
    print(f"runset labelled: {time.perf_counter() - t0:.3f} s")

    print("generating dataset defs...")
    t0 = time.perf_counter()
    gen_dataset_def(path_to_runset, label_dir_name=args.label_dir_name)
    print(f"dataset defs generated: {time.perf_counter() - t0:.3f} s")

    print("generating tasks files for Label Studio...")

    parasite_data_runset_path = Path(
        "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/"
    )

    t0 = time.perf_counter()
    generate_tasks_for_runset(
        labelled_run_paths,
        parasite_data_runset_path,
        label_dir_name=args.label_dir_name,
        tasks_file_name=args.tasks_file_name,
    )
    print(f"tasks files generated: {time.perf_counter() - t0:.3f} s")
    print("all done!")
