#! /usr/bin/env python3

import sys
import time

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict

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


def convert_labels_folder_class_ordering(
    run_set_path: Path,
    from_classes: List[str],
    to_classes: List[str],
    label_dir_name: str = "labels",
) -> Path:
    """If we encounter a folder that has a different ordering of classes
    than our master ordering, we must correct it.

    This function will go through every label and swap it for the correct one.

    It will return a path to the new, corrected labels.
    """
    # first, we need to make sure that from_classes is a strict subset of to_classes
    if set(from_classes) - set(to_classes) != set():
        raise ValueError(
            f"there are classes in from_classes that are not in to_classes; conversion failure "
            f"from_classes = {from_classes}, to_classes = {to_classes}, path = {run_set_path}"
        )

    if not (run_set_path / label_dir_name).exists():
        raise ValueError(f"director {run_set_path / label_dir_name} doesn't exist")

    (run_set_path / "converted_labels").mkdir(exist_ok=True)

    for file in (run_set_path / label_dir_name).iterdir():
        with open(file, "r") as f:
            contents = f.read().strip()

        # can probably do a r+ above, but I don't want to deal with seeking and
        # those complications - just reopen the file
        with open(run_set_path / "converted_labels" / file.name, "w") as g:
            for line in contents.split("\n"):
                contents = line.split(" ")

                if contents[0].isnumeric():
                    class_index = int(contents[0])
                    to_class = to_classes.index(from_classes[class_index])
                else:
                    class_index = contents[0]
                    to_class = to_classes.index(class_index)

                g.write(" ".join([str(to_class), *contents[1:]]) + "\n")

    with open("old_classes.txt", "w") as f:
        for class_id in from_classes:
            f.write(f"{class_id}\n")

    with open(run_set_path / "classes.txt", "w") as f:
        for class_id in to_classes:
            f.write(f"{class_id}\n")

    return run_set_path / "converted_labels"


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
        classes_path = folder_path / "classes.txt"

        if not (images_path.exists() and label_path.exists()):
            print(
                f"WARNING: image path or label path doesn't exist: {images_path}, {label_path}. Continuing..."
            )
            continue

        classes = (
            class_names_from_classes_dot_txt(classes_path)
            if classes_path.exists()
            else CLASSES  # if no classes.txt exists, assume that it has been default-labelled healthy
        )

        if CLASSES != classes:
            print(f"converting {folder_path} from {classes} to {CLASSES}...")
            t0 = time.perf_counter()
            label_path = convert_labels_folder_class_ordering(
                folder_path, classes, CLASSES, label_dir_name=label_dir_name
            )
            print(f"converted ({time.perf_counter() - t0:.3f} s)")

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
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to runset>")
        sys.exit(1)

    path_to_runset = Path(sys.argv[1])

    if not path_to_runset.exists():
        raise ValueError(f"{str(path_to_runset)} doesn't exist")

    gen_dataset_def(path_to_runset, verbose=True)
