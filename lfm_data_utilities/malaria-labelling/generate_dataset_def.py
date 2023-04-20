#! /usr/bin/env python3

import sys

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from ruamel import yaml

from labelling_constants import CLASSES

"""
This file will scan through the labeled data and create the data set definition file.
Here is an example format!

```
class_names: ["healthy", "ring", "schizont", "troph"]
dataset_split_fractions:
  train: 0.70
  test:  0.25
  val:   0.05
dataset_paths:
  cellpose_sample:
    image_path: /tmp/training_data_sample_2022_11_01_cyto2/sample_2022_11_01
    label_path: /tmp/training_data_sample_2022_11_01_cyto2/sample_2022_11_01_labels_cyto2
```
"""


def class_names_from_classes_dot_txt(path_to_classes_dot_txt: Path) -> List[str]:
    with open(str(path_to_classes_dot_txt), "r") as f:
        return [s.strip() for s in f.readlines() if s != ""]


def gen_dataset_def(
    path_to_runset_folder: Path, label_dir_name="labels", verbose=False
):
    folders = [Path(p).parent for p in path_to_runset_folder.glob("./**/images")]

    dataset_paths: Dict[str, Dict[str, str]] = dict()

    iterator = tqdm(enumerate(folders)) if verbose else enumerate(folders)
    for i, folder_path in iterator:
        # check classes
        images_path = folder_path / "images"
        label_path = folder_path / label_dir_name

        if not (images_path.exists() and label_path.exists()):
            print(
                f"WARNING: image path or label path doesn't exist: {images_path}, {label_path}. Continuing..."
            )
            continue

        dataset_paths[folder_path.name] = {
            "image_path": str(images_path),
            "label_path": str(label_path),
        }

    dataset_defs = {
        "class_names": CLASSES,
        "dataset_split_fractions": {"train": 0.75, "test": 0.20, "val": 0.05},
        "dataset_paths": dataset_paths,
    }

    yml = yaml.YAML()
    yml.indent(mapping=5, sequence=5, offset=3)

    with open("dataset_defs.yml", "w") as f:
        yml.dump(dataset_defs, f)
        print("dumped to dataset_defs.yml")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Dataset Definition Tool")

    generate_subparser = parser.add_subparser("generate")
    verify_subparser = parser.add_subparser("verify")

    generate_subparser.add_argument("path_to_runset", type=Path, help="Path to runset folder")
    verify_subparser.add_argument("path_to_dataset_defn_file", type=Path, help="Path to dataset definition file")

    args = parser.parse_args()

    if not path_to_runset.exists():
        raise ValueError(f"{str(path_to_runset)} doesn't exist")

    if args.subparser == "generate":
        gen_dataset_def(path_to_runset, verbose=True)
    elif args.subparser == "verify":
        try:
            from yogo.data.dataloader import load_dataset_description, InvalidDatasetDescriptionFile
        except ImportError:
            print("yogo is not installed. Please install yogo to verify dataset description file")
            sys.exit(1)

        # if InvalidDatasetDescriptionFile is raised, then the file is invalid
        try:
            load_dataset_description(args.path_to_dataset_defn_file)
        except InvalidDatasetDescriptionFile as e:
            print(f"Invalid dataset description file: {e}")
            sys.exit(1)
