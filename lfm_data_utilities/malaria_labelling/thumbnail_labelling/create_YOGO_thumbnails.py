import torch
import argparse

from tqdm import tqdm
from pathlib import Path

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
    args = parser.parse_args()

    device = choose_device()

    model, cfg = YOGO.from_pth(Path(args.path_to_pth), inference=True)
    model.to(device)
    model.eval()

    image_dataset = get_dataset(
        path_to_images=args.path_to_images,
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

    for batch in tqdm(image_dataloader):
        images, image_paths = batch
        images = images.to(device)

        with torch.no_grad():
            predictions = model(images)

        for image_path, prediction in zip(image_paths, predictions):
            tasks_file_writer.add_prediction(
                image_path,
                convert_formatted_YOGO_to_list(
                    format_preds(
                        prediction,
                        obj_thresh=args.obj_thresh,
                        iou_thresh=args.iou_thresh,
                    )
                )
            )

    tasks_file_writer.write(Path("tasks.json"))
