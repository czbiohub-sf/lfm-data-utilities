#! /usr/bin/env python3

import torch
import wandb
import argparse

from pathlib import Path

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path", type=Path)
    parser.add_argument("dataset_defn_path", type=Path)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "log to wandb - this will create a new run. If neither this nor "
            "--wandb-resume-id are provided, the run will be saved to a new folder"
        ),
    )
    parser.add_argument(
        "--wandb-resume-id",
        type=str,
        default=None,
        help=(
            "wandb run id - this will essentially append the results to an "
            "existing run, given by this run id"
        ),
    )
    parser.add_argument(
        "--dump-to-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "dump results to disk as a pkl file"
         )
    )
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

    num_workers = choose_dataloader_num_workers(len(test_dataset))  # type: ignore

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

    if args.wandb or args.wandb_resume_id:
        wandb.init(
            project="yogo",
            entity="bioengineering",
            config=config,
            notes="testing",
            tags=("test",),
            id=args.wandb_resume_id,
            resume="must" if args.wandb_resume_id else "allow",
        )

    test_metrics = Trainer._test(
        test_dataloader,
        "cuda",
        config,
        y,
    )

    if args.wandb or args.wandb_resume_id:
        Trainer._log_test_metrics(*test_metrics)

    if args.dump_to_disk:
        import pickle
        pickle.dump(test_metrics, open("test_metrics.pkl", "wb"))
