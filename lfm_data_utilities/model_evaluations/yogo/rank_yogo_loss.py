#! /usr/bin/env python3

"""
Goal: Run this script with YOGO over all of our labelled data and rank by loss!

- rank by weighted loss
- rank by loss type (classification v. IoU v. objectness)
"""

import os
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from pathlib import Path

from typing import Any, List, Dict, Union, Tuple, Optional, MutableMapping

from torch.utils.data import Dataset, ConcatDataset, DataLoader

from yogo.model import YOGO
from yogo.yogo_loss import YOGOLoss
from yogo.data.blobgen import BlobDataset
from yogo.data.dataloader import split_dataset
from yogo.data.dataset_description_file import load_dataset_description
from yogo.data.dataset import ObjectDetectionDataset, label_file_to_tensor


class ObjectDetectionDatasetWithPaths(ObjectDetectionDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        image_path = str(self._image_paths[index], encoding="utf-8")
        label_path = str(self._label_paths[index], encoding="utf-8")
        image = self.loader(image_path)
        labels = label_file_to_tensor(
            Path(label_path), self.Sx, self.Sy, self.notes_data
        )
        if self.normalize_images:
            # turns our torch.uint8 tensor 'sample' into a torch.FloatTensor
            image = image / 255
        return image, labels, image_path, label_path


def get_dataset(
    dataset_description_file: str,
    Sx: int,
    Sy: int,
    normalize_images: bool = False,
) -> Dataset[Any]:
    dataset_description = load_dataset_description(dataset_description_file)
    # can we speed this up? multiproc dataset creation?
    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDatasetWithPaths(
            dsp["image_path"],
            dsp["label_path"],
            Sx,
            Sy,
            normalize_images=normalize_images,
        )
        for dsp in tqdm(dataset_description.dataset_paths, desc="loading datasets")
    )
    return full_dataset


def collate_batch(batch):
    inputs, labels, image_paths, label_paths = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.stack(labels)
    return batched_inputs, batched_labels, image_paths, label_paths


def get_dataloader(
    dataset_descriptor_file: str,
    batch_size: int,
    Sx: int,
    Sy: int,
    normalize_images: bool = False,
) -> DataLoader:
    full_dataset = get_dataset(
        dataset_descriptor_file,
        Sx,
        Sy,
        normalize_images=normalize_images,
    )

    num_workers = min(len(os.sched_getaffinity(0)) // 2, 32)

    d = DataLoader(
        full_dataset,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        batch_size=batch_size,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        # optimal # of workers? >= 32
        num_workers=num_workers,  # type: ignore
        generator=torch.Generator().manual_seed(111111),
        collate_fn=collate_batch,
    )
    return d


def get_loss_df(dataset_descriptor_file, path_to_pth) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Y_loss = YOGOLoss(
        no_obj_weight=1,
        iou_weight=1,
        classify_weight=1,
        label_smoothing=0,
    ).to(device)

    net, net_cfg = YOGO.from_pth(path_to_pth)
    net.to(device)

    dataloader = get_dataloader(
        dataset_descriptor_file,
        batch_size=1,
        Sx=net.Sx,
        Sy=net.Sy,
        normalize_images=net_cfg["normalize_images"],
    )

    values: List[Dict[str, Union[int, float]]] = []

    for batch in tqdm(dataloader, desc="calculating loss"):
        inputs, labels, image_paths, label_paths = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss, loss_components = Y_loss(net(inputs), labels)
        values.append(
            {
                "image_path": image_paths.pop(),
                "label_path": label_paths.pop(),
                "total_loss": loss.item(),
                **loss_components,
            }
        )

    return pd.DataFrame(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rank YOGO loss", allow_abbrev=False)
    parser.add_argument(
        "dataset_descriptor_file",
        type=Path,
        help="path to yml dataset descriptor file",
    )
    parser.add_argument(
        "path_to_pth",
        type=Path,
        help="path to pth file",
    )
    args = parser.parse_args()
    get_loss_df(args.dataset_descriptor_file, args.path_to_pth).to_csv("loss.csv")
