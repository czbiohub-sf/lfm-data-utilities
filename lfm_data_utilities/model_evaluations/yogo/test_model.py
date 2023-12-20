#! /usr/bin/env python3

import os
import torch
import wandb
import argparse

from pathlib import Path

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from yogo.model import YOGO
from yogo.train import Trainer
from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.yogo_dataloader import (
    get_dataloader,
    choose_dataloader_num_workers,
    collate_batch,
)
from yogo.utils import (
    get_free_port,
)


def test_model(rank: int, world_size: int, args: argparse.Namespace) -> None:
    y, cfg = YOGO.from_pth(args.pth_path, inference=False)
    y.to("cuda")

    dataloaders = get_dataloader(
        args.dataset_defn_path,
        64,
        y.get_grid_size()[0],
        y.get_grid_size()[1],
        normalize_images=cfg["normalize_images"],
    )

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # take test and val since neither have been trained on and more data == gooder
    test_dataset: Dataset[ObjectDetectionDataset] = ConcatDataset(
        [dataloaders["val"].dataset, dataloaders["test"].dataset]
    )
    DistributedSampler(
        test_dataset,
        rank=rank,
        num_replicas=world_size,
    )

    num_workers = choose_dataloader_num_workers(len(test_dataset))  # type: ignore

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=64,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(111111),
        collate_fn=collate_batch,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    config = {
        "class_names": YOGO_CLASS_ORDERING,
        "no_classify": False,
        "iou_weight": 1,
        "healthy_weight": 1,
        "no_obj_weight": 0.5,
        "label_smoothing": 0.0001,
        "half": True,
        "model": args.pth_path,
        "test_set": args.dataset_defn_path,
    }

    wandb.init(
        project="yogo",
        entity="bioengineering",
        config=config,
        notes="testing",
        tags=("test",),
    )

    test_metrics = Trainer._test(
        test_dataloader,
        "cuda",
        config,
        y,
    )
    Trainer._log_test_metrics(*test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path", type=Path)
    parser.add_argument("dataset_defn_path", type=Path)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError(
            "at least 1 gpu is required for testing (otherwise it's painfully slow); "
            "if cpu training is required, we can add it back"
        )

    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = str(get_free_port())
    mp.spawn(test_model, args=(world_size, args), nprocs=world_size, join=True)
