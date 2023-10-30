#! /usr/bin/env python3

import torch
import wandb
import argparse

from pathlib import Path
from functools import partial

from torch.utils.data import Dataset, ConcatDataset, DataLoader

from yogo.model import YOGO
from yogo.train import Trainer
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.yogo_dataloader import get_dataloader, choose_dataloader_num_workers, collate_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path", type=Path)
    parser.add_argument("dataset_defn_path", type=Path)
    args = parser.parse_args()


    y, cfg = YOGO.from_pth(args.pth_path, inference=False)
    y.to("cuda")

    dataloaders = get_dataloader(
        args.dataset_defn_path,
        16,
        y.get_grid_size()[0],
        y.get_grid_size()[1],
        normalize_images=cfg["normalize_images"],
    )

    # take test and val since neither have been trained on and more data == gooder
    test_dataset: Dataset[ObjectDetectionDataset] = ConcatDataset(
        [dataloaders["val"].dataset, dataloaders["test"].dataset]
    )

    num_workers = choose_dataloader_num_workers(len(test_dataset))
    num_workers = 0

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=16,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(111111),
        collate_fn=collate_batch,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


    config = {
        "class_names": range(7),
        "no_classify": False,
        "iou_weight": 1,
        "healthy_weight": 1,
        "no_obj_weight": 0.5,
        "label_smoothing": 0.0001,
        "half": False,
    }

    wandb.init(
        project="yogo",
        entity="bioengineering",
        config=config,
        notes=f"testing",
        tags=("test",),
    )

    Trainer._test(
        test_dataloader,
        "cuda",
        config,
        y,
    )
