#! /usr/bin/env python3

import shutil
import argparse

from ruamel.yaml import YAML

from pathlib import Path

from yogo.utils import Timer
from yogo.data.dataset_definition_file import DatasetDefinition

"""
In my opinion, Ultralytic's dataset handling is clunky. We need to copy a bunch of data
into the datastructure they expect.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_description_file", help="Path to dataset description file", type=Path
    )
    parser.add_argument(
        "--output_dataset_path",
        help="Path to copied dataset (defaults to /hpc/projects/group.bioengineering/LFM_scope/labelled_yolo_data",
        type=Path,
        default=Path("/hpc/projects/group.bioengineering/LFM_scope/labelled_yolo_data"),
    )
    args = parser.parse_args()

    args.output_dataset_path.mkdir(parents=True, exist_ok=True)
    (args.output_dataset_path / "images").mkdir(exist_ok=True)
    (args.output_dataset_path / "labels").mkdir(exist_ok=True)

    with Timer("loading dataset description"):
        dataset_description = DatasetDefinition.from_yaml(args.dataset_description_file)

    with Timer("copying training dataset"):
        (args.output_dataset_path / "images" / "train").mkdir(exist_ok=True)
        (args.output_dataset_path / "labels" / "train").mkdir(exist_ok=True)
        (args.output_dataset_path / "images" / "val").mkdir(exist_ok=True)
        (args.output_dataset_path / "labels" / "val").mkdir(exist_ok=True)
        (args.output_dataset_path / "images" / "test").mkdir(exist_ok=True)
        (args.output_dataset_path / "labels" / "test").mkdir(exist_ok=True)

        for image_and_label_dir_dict in dataset_description.dataset_paths:
            image_dir, label_dir = (
                image_and_label_dir_dict.image_path,
                image_and_label_dir_dict.label_path,
            )
            for label in label_dir.glob("*.txt"):
                run_name = image_dir.parent.name
                new_label_name = f"{run_name}_{label.name}"
                new_image_name = f"{run_name}_{label.stem}.png"
                shutil.copy(
                    label,
                    args.output_dataset_path / "labels" / "train" / new_label_name,
                )
                with open(
                    args.output_dataset_path / "labels" / "train" / new_label_name, "w"
                ) as f:
                    for line in label.read_text().splitlines():
                        cls, x, y, w, h = line.split()
                        f.write(f"{int(cls) - 1} {x} {y} {w} {h}\n")
                shutil.copy(
                    image_dir / (label.with_suffix(".png")).name,
                    args.output_dataset_path / "images" / "train" / new_image_name,
                )

        for image_and_label_dir_dict in dataset_description.test_dataset_paths:
            image_dir, label_dir = (
                image_and_label_dir_dict.image_path,
                image_and_label_dir_dict.label_path,
            )
            for label in label_dir.glob("*.txt"):
                run_name = label.parents[2].name
                new_label_name = f"{run_name}_{label.name}"
                new_image_name = f"{run_name}_{label.stem}.png"
                with open(
                    args.output_dataset_path / "labels" / "val" / new_label_name, "w"
                ) as f:
                    for line in label.read_text().splitlines():
                        cls, x, y, w, h = line.split()
                        f.write(f"{int(cls) - 1} {x} {y} {w} {h}\n")
                shutil.copy(
                    image_dir / (label.with_suffix(".png")).name,
                    args.output_dataset_path / "images" / "val" / new_image_name,
                )

    with Timer("writing dataset description"):
        yaml = YAML()
        yolo_desc = {
            "path": str(args.output_dataset_path),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(dataset_description.classes)),
        }
        yaml.dump(yolo_desc, args.output_dataset_path / "dataset_description.yaml")
