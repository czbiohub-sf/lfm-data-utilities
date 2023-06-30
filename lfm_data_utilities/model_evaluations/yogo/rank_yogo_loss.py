#! /usr/bin/env python3

"""
Goal: Run this script with YOGO over all of our labelled data and rank by loss!

- rank by weighted loss
- rank by loss type (classification v. IoU v. objectness)
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw

from typing import Any, List, Dict, Union, Tuple

from torch.utils.data import Dataset, ConcatDataset, DataLoader

from yogo.model import YOGO
from yogo.yogo_loss import YOGOLoss
from yogo.utils.utils import bbox_colour
from yogo.utils import draw_yogo_prediction
from yogo.data.dataset_description_file import load_dataset_description
from yogo.data.dataset import (
    ObjectDetectionDataset,
    YOGO_CLASS_ORDERING,
    label_file_to_tensor,
    load_labels,
)


class YOGOPerLabelLoss(YOGOLoss):
    def forward(
        self, pred_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, loss_components = super().forward(pred_batch, label_batch)
        num_labels = max(label_batch[:, 0:1, :, :].sum().item(), 1)
        loss /= num_labels
        for k, v in loss_components.items():
            loss_components[k] = v / num_labels
        return loss, loss_components


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
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=batch_size,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        num_workers=num_workers,  # type: ignore
        generator=torch.Generator().manual_seed(111111),
        collate_fn=collate_batch,
    )
    return d


def get_loss_df(dataset_descriptor_file, path_to_pth) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Y_loss = YOGOPerLabelLoss(
        no_obj_weight=1,
        iou_weight=1,
        classify_weight=1,
        label_smoothing=0,
    ).to(device)

    net, net_cfg = YOGO.from_pth(path_to_pth, inference=False)
    net.to(device)
    net.eval()

    dataloader = get_dataloader(
        dataset_descriptor_file,
        batch_size=1,
        Sx=net.Sx,
        Sy=net.Sy,
        normalize_images=net_cfg["normalize_images"],
    )

    values: List[Dict[str, Union[int, float]]] = []

    for i, batch in enumerate(tqdm(dataloader, desc="calculating loss")):
        inputs, labels, image_paths, label_paths = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss, loss_components = Y_loss(net(inputs), labels)
        values.append(
            {
                "idx": i,
                "image_path": image_paths.pop(),
                "label_path": label_paths.pop(),
                "total_loss": loss.item(),
                **loss_components,
            }
        )

    return pd.DataFrame(values)


def select_top_n_paths(
    col_name: str, n: int, df: pd.DataFrame, ascending=False, include_index=False
) -> List[Tuple[int, str, str]]:
    return [
        (int(row[0]), row[1], row[2])
        for row in df.sort_values(by=col_name, ascending=ascending)[:n][
            ["image_path", "label_path"]
        ].itertuples(index=include_index)
    ]


def display_preds_and_labels(idx: int, df: pd.DataFrame, path_to_pth: str):
    image_path, label_path = df.iloc[idx][["image_path", "label_path"]]

    net, net_cfg = YOGO.from_pth(path_to_pth)

    image = torch.from_numpy(np.array(Image.open(image_path).convert("L")))
    image = image.unsqueeze(0).unsqueeze(0).float()
    if net_cfg["normalize_images"]:
        image = image / 255

    notes_path = Path(label_path).parent.parent / "notes.json"
    if notes_path.exists():
        with open(notes_path) as f:
            json_data = json.load(f)
    else:
        json_data = None

    label_tensor = load_labels(label_path, notes_data=json_data)

    pred = net(image)[0, ...]

    prediction_bbox_img = draw_yogo_prediction(
        image,
        pred,
        labels=YOGO_CLASS_ORDERING,
        images_are_normalized=net_cfg["normalize_images"],
    )

    draw = ImageDraw.Draw(prediction_bbox_img)

    img_h, img_w = image.shape[-2:]
    for r in label_tensor:
        # convert cxcywh to xyxy and scale to image size
        x1 = (r[1] - r[3] / 2) * img_w
        y1 = (r[2] - r[4] / 2) * img_h
        x2 = (r[1] + r[3] / 2) * img_w
        y2 = (r[2] + r[4] / 2) * img_h
        label = YOGO_CLASS_ORDERING[int(r[0])]
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=bbox_colour(label),
        )
        draw.text((x1, y2), f"label: {label}", (0, 0, 0, 255))

    return prediction_bbox_img


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
