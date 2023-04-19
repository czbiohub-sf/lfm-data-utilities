#! /usr/bin/env python3

import sys
import signal

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Generator, Iterable, Tuple, List, Optional, Dict, Any

from yogo.utils import draw_rects
from yogo.data.dataset import read_grayscale, load_labels

from labelling_constants import CLASSES

signal.signal(signal.SIGINT, signal.SIG_DFL)


def find_label_file(label_dir: Path, image_path: Path) -> Path:
    extensions = (".txt", ".csv", ".tsv", "")
    for ext in extensions:
        label_path = label_dir / image_path.with_suffix(ext).name
        if label_path.exists():
            return label_path

    raise FileNotFoundError(f"label file not found for {str(image_path)}")


def plot_img_labels_pair(
    image_path: Path, label_path: Path, notes_file: Optional[Dict[str, Any]]
):
    labels = load_labels(label_path, dataset_classes=CLASSES)

    img = read_grayscale(str(image_path)).squeeze()
    annotated_img = draw_rects(img, labels, labels=CLASSES)

    plt.imshow(annotated_img)
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


def load_notes_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("visualize some labels in a run")
    parser.add_argument("path_to_run", type=Path, help="path to run")
    parser.add_argument(
        "--label-dir-name",
        type=str,
        default="labels",
        help="name of label dir - e.g. yogo_labels. Defaults to 'labels'",
    )

    args = parser.parse_args()

    image_dir = args.path_to_run / "images"
    label_dir = args.path_to_run / args.label_dir_name
    notes = load_notes_file(args.path_to_run / "notes.json")

    print("ctrl-c to stop")
    for image_path, label_path in make_img_label_pairs(image_dir, label_dir):
        plot_img_labels_pair(image_path, label_path, notes)
