#! /usr/bin/env python3

"""
Goal: Run this script with YOGO over all of our labelled data and rank by loss!

- rank by weighted loss
- rank by loss type (classification v. IoU v. objectness)
"""


from typing import Tuple

from pathlib import Path

import os
import torch

from tqdm import tqdm

from torch.utils.data import Dataset, ConcatDataset, DataLoader

from typing import Dict, Tuple, Optional, Any, MutableMapping

from yogo.data.blobgen import BlobDataset
from yogo.data.dataset import ObjectDetectionDataset
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


def get_datasets(
    dataset_description_file: str,
    Sx: int,
    Sy: int,
    split_fractions_override: Optional[Dict[str, float]] = None,
    normalize_images: bool = False,
) -> MutableMapping[str, Dataset[Any]]:
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
        for dsp in tqdm(dataset_description.dataset_paths)
    )

    if dataset_description.test_dataset_paths is not None:
        test_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
            ObjectDetectionDatasetWithPaths(
                dsp["image_path"],
                dsp["label_path"],
                Sx,
                Sy,
                normalize_images=normalize_images,
            )
            for dsp in tqdm(dataset_description.test_dataset_paths)
        )
        split_datasets: MutableMapping[str, Dataset[Any]] = {
            "train": full_dataset,
            **split_dataset(test_dataset, dataset_description.split_fractions),
        }
    else:
        split_datasets = split_dataset(
            full_dataset, dataset_description.split_fractions
        )

    # hardcode the blob agumentation for now
    # this should be moved into the dataset description file
    if dataset_description.thumbnail_augmentation is not None:
        # some issue w/ Dict v Mapping TODO come back to this
        bd = BlobDataset(
            dataset_description.thumbnail_augmentation,  # type: ignore
            Sx=Sx,
            Sy=Sy,
            n=8,
            length=len(split_datasets["train"]) // 4,  # type: ignore
            blend_thumbnails=True,
            thumbnail_sigma=2,
            normalize_images=normalize_images,
        )
        split_datasets["train"] = ConcatDataset([split_datasets["train"], bd])

    return split_datasets


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
    training: bool = True,
    preprocess_type: Optional[str] = None,
    vertical_crop_size: Optional[float] = None,
    resize_shape: Optional[Tuple[int, int]] = None,
    split_fractions_override: Optional[Dict[str, float]] = None,
    normalize_images: bool = False,
) -> Dict[str, DataLoader]:
    split_datasets = get_datasets(
        dataset_descriptor_file,
        Sx,
        Sy,
        split_fractions_override=split_fractions_override,
        normalize_images=normalize_images,
    )

    num_workers = min(len(os.sched_getaffinity(0)) // 2, 32)

    d = dict()
    for designation, dataset in split_datasets.items():
        d[designation] = DataLoader(
            dataset,
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
