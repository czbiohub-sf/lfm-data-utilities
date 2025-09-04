#! /usr/bin/env python3


"""
We're training on a subset of images from many Uganda sets.

Specfically,
    https://github.com/czbiohub-sf/lfm-dataset-definitions/blob/main/dataset-subsets/uganda-thumbnail-corrections.yml

Previously we copied the selected images, but we are now moving them. So, we need to go
back and delete the images that were copied.

We'll use `uganda-thumbnail-corrections.yml` / yogo.data.dataset_definition_file to to
through each set of images.
"""

import re
import argparse

from pathlib import Path

from yogo.data.dataset_definition_file import DatasetDefinition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    defn = DatasetDefinition.from_yaml(
        "biohub-labels/dataset_defs/dataset-subsets/uganda-healthy.yml"
    )

    for dset in defn.all_dataset_paths:
        parent_dir = Path(dset.label_path.parents[2])

        run_dir = next(
            p
            for p in parent_dir.rglob(dset.label_path.parent.name)
            if (
                (re.match(r".*Uganda_full_?\d?_images", str(p)) is not None)
                and all(
                    s not in str(p) for s in ("dense-data", "subsets-for-labelling")
                )
            )
        )

        print(run_dir)

        # find the correct label dir
        if (run_dir / "yogo_labels").exists():
            label_dir = run_dir / "yogo_labels"
        elif (run_dir / "labels").exists():
            label_dir = run_dir / "labels"
        else:
            print(
                f"Could not find {str(run_dir / 'labels')} "
                f"or {str(run_dir / 'yogo_labels')}"
            )
            continue

        n_images, n_labels = 0, 0
        for img in dset.image_path.glob("*.png"):
            if args.execute:
                try:
                    (run_dir / "images" / img.name).unlink()
                except FileNotFoundError:
                    print(f"Could not find {str(run_dir / 'images' / img.name)}")

            n_images += int((run_dir / "images" / img.name).exists())

        # delete the labels
        for label in dset.label_path.glob("*.txt"):
            if args.execute:
                # annoyingly, our labels are either in yogo_labels or labels
                try:
                    (label_dir / label.name).unlink()
                except FileNotFoundError:
                    print(f"Could not find {str(label_dir / label.name)}")

            n_labels += int((label_dir / label.name).exists())

        print(
            f"{'deleted' if args.execute else 'skipped'} {n_images} images and "
            f"{n_labels} labels from {str(dset.label_path)}"
        )
