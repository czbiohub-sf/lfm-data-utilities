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

from lfm_data_utilities.malaria_labelling.thumbnail_labelling.sort_thumbnails import (
    sort_thumbnails,
)
from lfm_data_utilities.malaria_labelling.thumbnail_labelling.create_thumbnails import (
    create_thumbnails_for_sorting,
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
        )
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
        create_thumbnails_for_sorting(
            args.path_to_output_dir,
            args.path_to_labelled_data_ddf,
            args.overwrite_previous_thumbnails,
            args.ignore_class,
        )
    else:
        parser.print_help()
