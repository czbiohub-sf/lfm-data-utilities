#! /usr/bin/env python3

import torch
import argparse
import warnings

import numpy as np
import numpy.typing as npt

from tqdm import tqdm
from pathlib import Path
from functools import partial

from torch.utils.data import ConcatDataset, DataLoader, Subset

from yogo.model import YOGO
from yogo.train import Trainer
from yogo.utils import choose_device
from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.yogo_dataloader import choose_dataloader_num_workers, collate_batch
from yogo.data.data_transforms import DualInputId
from yogo.data.dataset_description_file import load_dataset_description
from yogo.utils.default_hyperparams import DefaultHyperparams as df

# set seeds so we can reproduce + if we do this w/ slurm arrays,
# each job could pick a separate fold for calculation
torch.manual_seed(7271978)
np.random.seed(7271978)

""" The purpose of this file is to try to understand the stability of the YOGO confusion matrix.

Given a large testing dataset, split it into N "folds" and test on each. Then, using the testing
results of each, compute a bunch of statistics on the testing results. The idea being that if, e.g.
standard deviation for the confusion matrix is really high, then we can't trust it to be stable.
"""


def load_description_to_dataloader(
    dataset_description_file: Path,
    Sx: int,
    Sy: int,
    num_folds: int,
    normalize_images: bool,
) -> list[DataLoader]:
    dataset_descriptor = load_dataset_description(str(dataset_description_file))
    all_dataset_paths = dataset_descriptor.dataset_paths + (
        dataset_descriptor.test_dataset_paths or []
    )

    dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dsp["image_path"],
            dsp["label_path"],
            Sx,
            Sy,
            normalize_images=normalize_images,
        )
        for dsp in tqdm(all_dataset_paths)
    )

    dataset_indicies = np.arange(len(dataset))
    dataset_splits = np.array_split(dataset_indicies, num_folds)
    num_workers = choose_dataloader_num_workers(len(dataset), 8)

    return [
        DataLoader(
            Subset(dataset, chunk_indicies),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            batch_size=64,
            num_workers=4,
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(7271978),
            collate_fn=partial(collate_batch, transforms=DualInputId()),
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
        for chunk_indicies in dataset_splits
    ]


def write_test_results(
    output_dir: Path,
    mean_loss,
    mAP,
    confusion_data,
    accuracy,
    roc_curves,
    precision,
    recall,
    calibration_error,
    class_names,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "confusion_data.npy", confusion_data.cpu().numpy())


def normalize_confusion_matrix(confusion_matrix: npt.NDArray) -> npt.NDArray:
    row_sum = confusion_matrix.sum(axis=1, keepdims=True)
    mat = np.divide(confusion_matrix, row_sum, where=row_sum != 0)
    assert np.isfinite(mat).all()
    return mat


if __name__ == "__main__":
    print("is this even starting like")
    parser = argparse.ArgumentParser(description="N-fold test of YOGO")
    parser.add_argument(
        "pth_path",
        type=Path,
        help="path to .pth file",
    )
    parser.add_argument(
        "dataset_descriptor_file",
        type=str,
        help="path to yml dataset descriptor file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="path to output directory - defaults to parent dir of pth_path",
        default=None,
    )
    parser.add_argument("-N", "--N", type=int, help="number of folds", default=5)
    args = parser.parse_args()

    output_dir = args.output_dir or args.pth_path.parent

    print("loading model...", end=" ")
    y_, cfg = YOGO.from_pth(args.pth_path)
    print("loaded")

    print("loading dataloaders...", end=" ")
    dataloaders = load_description_to_dataloader(
        args.dataset_descriptor_file, y_.Sx, y_.Sy, args.N, cfg["normalize_images"]
    )
    print("loaded")

    device = choose_device()
    if str(device) == "cpu":
        warnings.warn("no gpu found; this will be slow :(")

    # These are just some standard opts weeee
    config = {
        "class_names": YOGO_CLASS_ORDERING,
        "no_classify": False,
        "healthy_weight": df.HEALTHY_WEIGHT,
        "iou_weight": df.IOU_WEIGHT,
        "no_obj_weight": df.NO_OBJ_WEIGHT,
        "classify_weight": df.CLASSIFY_WEIGHT,
        "label_smoothing": df.LABEL_SMOOTHING,
        "half": False,
    }

    print("herewego")
    confusion_matricies = []
    for i, dataloader in tqdm(enumerate(dataloaders)):
        y, cfg = YOGO.from_pth(args.pth_path)
        y.eval()
        y.to(device)
        (
            mean_loss,
            mAP,
            confusion_data,
            accuracy,
            roc_curves,
            precision,
            recall,
            calibration_error,
            class_names,
        ) = Trainer._test(dataloader, device, config, y)

        write_test_results(
            output_dir / f"{i}",
            mean_loss,
            mAP,
            confusion_data,
            accuracy,
            roc_curves,
            precision,
            recall,
            calibration_error,
            class_names,
        )

        confusion_matricies.append(np.array(confusion_data.cpu().numpy()))

    normalized_matricies = [
        normalize_confusion_matrix(confusion_matrix)
        for confusion_matrix in confusion_matricies
    ]
    normalized_mean = np.mean(normalized_matricies, axis=0)
    normalized_std = np.std(normalized_matricies, axis=0, ddof=1)

    np.save(output_dir / "normalized_mean.npy", normalized_mean)
    np.save(output_dir / "normalized_std.npy", normalized_std)
