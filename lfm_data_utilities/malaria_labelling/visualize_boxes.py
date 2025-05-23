#! /usr/bin/env python3

import json
import signal
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Generator, Iterable, Tuple

from yogo.utils.utils import bbox_colour
from yogo.data.yogo_dataset import load_labels

from lfm_data_utilities import YOGO_CLASS_ORDERING

signal.signal(signal.SIGINT, signal.SIG_DFL)


def find_label_file(label_dir: Path, image_path: Path) -> Path:
    extensions = (".txt", ".csv", ".tsv", "")
    for ext in extensions:
        label_path = label_dir / image_path.with_suffix(ext).name
        if label_path.exists():
            return label_path

    raise FileNotFoundError(f"label file not found for {str(image_path)}")


def draw_rects(image_path, label_path):
    pil_img = Image.open(image_path).convert("L")

    notes_path = Path(label_path).parent.parent / "notes.json"
    if notes_path.exists():
        with open(notes_path) as f:
            json_data = json.load(f)
    else:
        json_data = None

    print(YOGO_CLASS_ORDERING)
    label_tensor = load_labels(
        label_path, notes_data=json_data, classes=YOGO_CLASS_ORDERING
    )

    rgb = Image.new("RGBA", pil_img.size)
    rgb.paste(pil_img)

    draw = ImageDraw.Draw(rgb)

    img_w, img_h = pil_img.size
    for r in label_tensor:
        # convert cxcywh to xyxy and scale to image size
        x1 = (r[1] - r[3] / 2) * img_w
        y1 = (r[2] - r[4] / 2) * img_h
        x2 = (r[1] + r[3] / 2) * img_w
        y2 = (r[2] + r[4] / 2) * img_h
        label = YOGO_CLASS_ORDERING[int(r[0])]
        label = YOGO_CLASS_ORDERING.index(label)
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=bbox_colour(label, num_classes=7),
        )
        draw.text((x1, y2), f"label: {label}", (0, 0, 0, 255))

    return rgb


def plot_img_labels_pair(image_path: Path, label_path: Path):
    annotated_img = draw_rects(image_path, label_path)
    plt.figure(figsize=(12 * 1.33, 12))
    plt.imshow(annotated_img)
    plt.tight_layout()
    plt.show()


def make_img_label_pairs(
    image_dir: Path, label_dir: Path
) -> Generator[Tuple[Path, Path], None, None]:
    image_iter: Iterable = image_dir.glob("*.png")

    for image_path in image_iter:
        try:
            label_path = find_label_file(label_dir, image_path)
            print(image_path, label_path)
            yield image_path, label_path
        except FileNotFoundError as e:
            print(f"no label file: {e}")
            print("continuing...")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("visualize some labels in a run")
    parser.add_argument("path_to_images", type=Path, help="path to run")
    parser.add_argument("path_to_labels", type=Path, help="path to labels")
    parser.add_argument(
        "--label-dir-name",
        type=str,
        default="labels",
        help="name of label dir - e.g. yogo_labels. Defaults to 'labels'",
    )

    args = parser.parse_args()

    image_dir = args.path_to_images
    label_dir = args.path_to_labels

    print("ctrl-c to stop")
    for image_path, label_path in make_img_label_pairs(image_dir, label_dir):
        plot_img_labels_pair(image_path, label_path)
