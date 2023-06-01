#! /usr/bin/env python3

import sys

from tqdm import tqdm
from pathlib import Path
from typing import Dict

from ruamel import yaml


"""
This file will scan through the labeled data and create the data set definition file.
Here is an example format!

```
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


def gen_dataset_def(
    path_to_runset_folder: Path,
    label_dir_name="labels",
    verbose=False,
    dataset_def_name="dataset_defs",
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
        "dataset_split_fractions": {"train": 0.75, "test": 0.20, "val": 0.05},
        "dataset_paths": dataset_paths,
    }

    yml = yaml.YAML()
    yml.indent(mapping=5, sequence=5, offset=3)

    with open(Path(dataset_def_name).with_suffix(".yml"), "w") as f:
        yml.dump(dataset_defs, f)
        print(f"dumped to {str(Path(dataset_def_name).with_suffix('.yml'))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Dataset Definition Tool")

    # one subparser for generating dataset definition file, one for verifying them
    subparsers = parser.add_subparsers(dest="subparser")
    generate_subparser = subparsers.add_parser("generate")
    verify_subparser = subparsers.add_parser("verify")

    generate_subparser.add_argument(
        "path_to_runset", type=Path, help="Path to runset folder"
    )
    generate_subparser.add_argument(
        "--label-dir-name",
        type=str,
        help="label dir name (defaults to `labels`)",
        default="labels",
    )
    generate_subparser.add_argument(
        "--dataset-def-name",
        type=str,
        help="dataset definition file name (defaults to `dataset_defs.yml`)",
        default="labels",
    )

    verify_subparser.add_argument(
        "path_to_dataset_defn_file",
        type=Path,
        help="Path to dataset definition file, or folder of the same",
    )

    args = parser.parse_args()

    if args.subparser == "generate":
        if not args.path_to_runset.exists():
            raise ValueError(f"{str(args.path_to_runset)} doesn't exist")

        gen_dataset_def(
            args.path_to_runset,
            verbose=True,
            label_dir_name=args.label_dir_name,
            dataset_def_name=args.dataset_def_name,
        )
    elif args.subparser == "verify":
        try:
            from yogo.data.dataloader import (
                load_dataset_description,
                InvalidDatasetDescriptionFile,
            )
        except ImportError:
            print(
                "yogo is not installed. Please install yogo to verify dataset description file"
            )
            sys.exit(1)

        # if InvalidDatasetDescriptionFile is raised, then the file is invalid
        # if path_to_dataset_defn_file is a directory, iterate over yml files and check all of them
        if args.path_to_dataset_defn_file.is_dir():
            for path in args.path_to_dataset_defn_file.glob("*.yml"):
                try:
                    load_dataset_description(path)
                except InvalidDatasetDescriptionFile as e:
                    print(f"{path} is invalid: {e}")
                    sys.exit(1)
        else:
            try:
                load_dataset_description(args.path_to_dataset_defn_file)
            except InvalidDatasetDescriptionFile as e:
                print(f"{args.path_to_dataset_defn_file} is invalid: {e}")
                sys.exit(1)
    else:
        parser.print_help()
