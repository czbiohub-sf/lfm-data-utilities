import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from lfm_data_utilities.malaria_labelling.label_studio_converter.create_ls_file import (
    LabelStudioTasksFile,
    convert_formatted_YOGO_to_list,
)

from yogo.model import YOGO
from yogo.utils.utils import format_preds
from yogo.utils.argparsers import unsigned_float
from yogo.data.image_path_dataset import get_dataset, collate_fn
from yogo.infer import choose_device, choose_dataloader_num_workers


def create_confidence_filtered_tasks_file_from_YOGO(
    path_to_pth: Path,
    path_to_images: Path,
    output_path: Optional[Path] = None,
    obj_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    min_class_confidence_thresh: Optional[float] = None,
    max_class_confidence_thresh: Optional[float] = None,
):
    device = choose_device()

    model, cfg = YOGO.from_pth(Path(path_to_pth), inference=True)
    model.to(device)
    model.eval()

    image_dataset = get_dataset(
        path_to_images=path_to_images,
        normalize_images=cfg["normalize_images"],
    )

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=choose_dataloader_num_workers(len(image_dataset)),
    )

    tasks_file_writer = LabelStudioTasksFile()

    for batch in tqdm(
        image_dataloader, desc=f"yogo inference on {path_to_images.parent.name}"
    ):
        images, image_paths = batch
        images = images.to(device)

        with torch.no_grad():
            predictions = model(images)

        for image_path, prediction in zip(image_paths, predictions):
            formatted_preds = format_preds(
                prediction,
                obj_thresh=obj_thresh,
                iou_thresh=iou_thresh,
            )
            formatted_pred_class_confidences = formatted_preds[:, 5:].max(dim=1).values
            formatted_pred_class_mask = torch.logical_and(
                (min_class_confidence_thresh or 0) <= formatted_pred_class_confidences,
                formatted_pred_class_confidences <= (max_class_confidence_thresh or 1),
            )
            formatted_preds = formatted_preds[formatted_pred_class_mask]

            if len(formatted_preds) > 0:
                tasks_file_writer.add_prediction(
                    image_path,
                    convert_formatted_YOGO_to_list(formatted_preds),
                )

    tasks_file_writer.write(output_path or Path("tasks.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_pth", type=Path, help="path to .pth file defining the model"
    )
    parser.add_argument(
        "path_to_images",
        type=Path,
    )
    parser.add_argument(
        "--obj-thresh",
        type=unsigned_float,
        default=0.5,
        help="objectness threshold for predictions (default 0.5)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=unsigned_float,
        default=0.5,
        help="intersection over union threshold for predictions (default 0.5)",
    )
    parser.add_argument(
        "--min-class-confidence-thresh",
        type=unsigned_float,
        default=None,
        help="minimum class confidence threshold for predictions (default None)",
    )
    parser.add_argument(
        "--max-class-confidence-thresh",
        type=unsigned_float,
        default=None,
        help="maximum class confidence threshold for predictions (default None)",
    )
    args = parser.parse_args()

    create_confidence_filtered_tasks_file_from_YOGO(
        args.path_to_pth,
        args.path_to_images,
        obj_thresh=args.obj_thresh,
        iou_thresh=args.iou_thresh,
        min_class_confidence_thresh=args.min_class_confidence_thresh,
        max_class_confidence_thresh=args.max_class_confidence_thresh,
    )
