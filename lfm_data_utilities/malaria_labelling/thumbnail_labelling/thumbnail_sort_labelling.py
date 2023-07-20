#! /usr/bin/env python3

""" Thumbnail Sort Labelling

High-level Goal:
- take dataset description file for
- create `tasks.json` file (files?)
- create folders of thumbnails (the 'thumbnail-folder'), sorted by class
    - thumbnail filenames should include class, run id, cell id
    - should also create 'target' folders in the same place!
    - create a file in the thumbnail folder that maps a run id to the path of the tasks.json file
- once sorted files are created,
    - go through folders, correcting labels in the tasks.json file(s?)
- zip and save biohub-labels/vetted as a backup
- re-export the tasks.json file(s)

Notes:
- once a run has cell classes that are overwritten, we can NOT re-export
that run from label studio, since it will no longer have correct files.
"""

import argparse

from pathlib import Path

from yogo.data import YOGO_CLASS_ORDERING

from lfm_data_utilities.malaria_labelling.thumbnail_labelling.sort_thumbnails import (
    sort_thumbnails,
)
from lfm_data_utilities.malaria_labelling.thumbnail_labelling.create_thumbnails import (
    create_thumbnails_from_tasks_maps,
    create_folders_for_output_dir,
    create_tasks_files_from_labels,
    create_confidence_filtered_tasks_from_YOGO,
)


DEFAULT_LABELS_PATH = Path(
    "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels/"
)


if __name__ == "__main__":
    try:
        boolean_action = argparse.BooleanOptionalAction  # type: ignore
    except AttributeError:
        boolean_action = "store_true"  # type: ignore

    default_ddf = (
        DEFAULT_LABELS_PATH
        / "dataset_defs"
        / "human-labels"
        / "all-labelled-data-train-only.yml"
    )

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser")

    create_thumbnails_parser = subparsers.add_parser("create-thumbnails")
    create_thumbnails_parser.add_argument("path_to_output_dir", type=Path)
    create_thumbnails_parser.add_argument(
        "--path-to-labelled-data-ddf",
        help=(
            "path to dataset descriptor file for labelled data - in general you should not need to change this, "
            f"since we mostly want to correct labels for human-labelled data (default {default_ddf})"
        ),
        default=default_ddf,
        type=Path,
    )
    create_thumbnails_parser.add_argument(
        "--overwrite-previous-thumbnails",
        action="store_true",
        help="if set, will overwrite previous thumbnails",
    )
    create_thumbnails_parser.add_argument(
        "--ignore-class",
        action="append",
        help=(
            "if set, will ignore this class when creating thumbnails - e.g. `--ignore-class healthy`\n"
            "you can provide this argument multiple times to ignore multiple classes - e.g. `--ignore-class healthy --ignore-class misc`\n"
            "suggested: `--ignore-class healthy`"
        ),
    )
    create_thumbnails_parser.add_argument(
        "--thumbnail-type",
        choices=["labels", "yogo-confidence", "yogo-incorrect"],
        default="labels",
        help=(
            "which type of thumbnail to create - labels: thumbnails with the labels as provided by the human labelers\n"
            "yogo-confidence: thumbnails predicted by YOGO, with high class confidence scores filtered by `--max-confidence`\n"
            "yogo-incorrect: thumbnails where the yogo model was incorrect"
        ),
    )
    create_thumbnails_parser.add_argument(
        "--path-to-pth",
        type=Path,
        help=(
            "if `--thumbnail-type yogo-confidence` or `--thumbnail-type yogo-incorrect` is provided, this is the path \n"
            "to the .pth file containing the model weights"
        ),
    )
    create_thumbnails_parser.add_argument(
        "--max-confidence",
        type=float,
        default=0.9,
        help=(
            "if `--thumbnail-type yogo-confidence` is provided, this is the maximum confidence score to include in the thumbnail"
        ),
    )

    sort_thumbnails_parser = subparsers.add_parser("sort-thumbnails")
    sort_thumbnails_parser.add_argument("path_to_thumbnails", type=Path)
    sort_thumbnails_parser.add_argument(
        "--commit",
        action=boolean_action,
        help="actually modify files on the file system instead of reporting what would be changed",
        default=False,
    )

    args = parser.parse_args()

    if args.subparser == "sort-thumbnails":
        if not args.commit:
            print(
                "--commit not provided, so this will be a dry run - no files will be modified"
            )

        sort_thumbnails(args.path_to_thumbnails, args.commit)
    elif args.subparser == "create-thumbnails":
        class_dirs, tasks_dir = create_folders_for_output_dir(
            args.path_to_output_dir,
            YOGO_CLASS_ORDERING,
            force_overwrite=args.overwrite_previous_thumbnails,
            ignore_classes=args.ignore_class or [],
        )

        if args.thumbnail_type == "labels":
            tasks_and_labels_paths = create_tasks_files_from_labels(
                args.path_to_labelled_data_ddf, tasks_dir
            )
            create_thumbnails_from_tasks_maps(
                args.path_to_output_dir,
                tasks_and_labels_paths,
                tasks_dir,
                class_dirs,
                classes_to_ignore=args.ignore_class or [],
            )
        elif args.thumbnail_type == "yogo-confidence":
            if args.path_to_pth is None:
                raise ValueError(
                    "if `--thumbnail-type yogo-confidence` is provided, `--path-to-pth` must also be provided"
                )
            tasks_and_labels_paths = create_confidence_filtered_tasks_from_YOGO(
                args.path_to_labelled_data_ddf,
                tasks_dir,
                args.path_to_pth,
                max_class_confidence_thresh=args.max_confidence,
            )
        elif args.thumbnail_type == "yogo-incorrect":
            if args.path_to_pth is None:
                raise ValueError(
                    "if `--thumbnail-type yogo-incorrect` is provided, `--path-to-pth` must also be provided"
                )
            raise NotImplementedError()
    else:
        parser.print_help()
