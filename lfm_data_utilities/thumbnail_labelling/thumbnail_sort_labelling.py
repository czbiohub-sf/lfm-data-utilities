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

import shutil
import logging
import argparse

from pathlib import Path

from lfm_data_utilities import YOGO_CLASS_ORDERING

from lfm_data_utilities.thumbnail_labelling.sort_thumbnails import (
    sort_thumbnails,
)

from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import (
    LFM_SCOPE_PATH,
    gen_task,
)


from lfm_data_utilities.thumbnail_labelling.yogo_filters import (
    create_confidence_filtered_tasks_file_from_YOGO,
    create_correctness_filtered_tasks_file_from_YOGO,
)

from lfm_data_utilities.thumbnail_labelling.create_thumbnails import (
    create_tasks_files_from_path_to_labelled_data_ddf,
    create_tasks_file_from_path_to_run,
    create_thumbnails_from_tasks_maps,
    create_folders_for_output_dir,
)


DEFAULT_LABELS_PATH = Path(
    "/hpc/projects/group.bioengineering/LFM_scope/biohub-labels/"
)

logging.getLogger("PIL").setLevel(logging.WARNING)  # quiet, you!


def main():
    try:
        boolean_action = argparse.BooleanOptionalAction  # type: ignore
    except AttributeError:
        boolean_action = "store_true"  # type: ignore

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser")

    create_thumbnails_parser = subparsers.add_parser("create")
    create_thumbnails_parser.add_argument("path_to_output_dir", type=Path)

    input_source = create_thumbnails_parser.add_mutually_exclusive_group()
    input_source.add_argument(
        "--path-to-labelled-data-ddf",
        help=("path to dataset descriptor file for labelled data"),
        type=Path,
    )
    input_source.add_argument(
        "--path-to-run",
        help="path to dataset descriptor file run",
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
        default=1.0,
        help=(
            "if `--thumbnail-type yogo-confidence` is provided, this is the "
            "maximum confidence score to include in the thumbnail (default 1)"
        ),
    )
    create_thumbnails_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "if `--thumbnail-type yogo-confidence` is provided, this is the "
            "minimum confidence score to include in the thumbnail (default 0)"
        ),
    )
    create_thumbnails_parser.add_argument(
        "--obj-thresh",
        "--obj-threshold",
        "--objectness-threshold",
        type=float,
        default=0.5,
        help="objectness threshold for YOGO predictions",
    )
    create_thumbnails_parser.add_argument(
        "--iou-thresh",
        "--iou-threshold",
        "--iou-threshold",
        type=float,
        default=0.5,
        help="iou threshold for YOGO predictions",
    )
    create_thumbnails_parser.add_argument(
        "--image-server-relative-parent-override",
        type=Path,
        default=LFM_SCOPE_PATH,
        help=(
            "override the image server relative parent for generating tasks; if you want the root of the image server"
            "to be different from LFM_Scope, you can provide that here - but don't touch this if that doesn't make sense"
        ),
    )

    sort_thumbnails_parser = subparsers.add_parser("sort")
    sort_thumbnails_parser.add_argument("path_to_thumbnails", type=Path)
    sort_thumbnails_parser.add_argument(
        "--commit",
        action=boolean_action,
        help="actually modify files on the file system instead of reporting what would be changed",
        default=False,
    )
    sort_thumbnails_parser.add_argument(
        "--ok-if-class-mismatch",
        action=boolean_action,
        help=(
            "if set, will not raise an error if the class of a thumbnail does not match the folder it is in"
        ),
        default=False,
    )

    args = parser.parse_args()

    if args.subparser == "sort":
        if not args.commit:
            print(
                "--commit not provided, so this will be a dry run - no files will be modified"
            )

        sort_thumbnails(args.path_to_thumbnails, args.commit, args.ok_if_class_mismatch)
    elif args.subparser == "create":
        if not args.overwrite_previous_thumbnails and args.path_to_output_dir.exists():
            raise RuntimeError(
                f"output folder {args.path_to_output_dir} exists; to overwrite, use --overwrite-previous-thumbnails"
            )

        class_dirs, tasks_dir = create_folders_for_output_dir(
            args.path_to_output_dir,
            YOGO_CLASS_ORDERING,
            force_overwrite=args.overwrite_previous_thumbnails,
            ignore_classes=args.ignore_class or [],
        )

        if args.thumbnail_type == "labels":

            def func(image_path: Path, label_path: Path, tasks_path: Path) -> None:
                gen_task(
                    folder_path=Path(label_path).parent,
                    relative_parent=args.image_server_relative_parent_override,
                    images_dir_path=image_path,
                    label_dir_name=Path(label_path).name,
                    tasks_path=tasks_path,
                )

        elif args.thumbnail_type == "yogo-confidence":
            if args.path_to_pth is None:
                raise ValueError(
                    "if `--thumbnail-type yogo-confidence` is provided, `--path-to-pth` must also be provided"
                )

            def func(image_path: Path, label_path: Path, tasks_path: Path) -> None:
                create_confidence_filtered_tasks_file_from_YOGO(
                    path_to_pth=args.path_to_pth,
                    path_to_images=image_path,
                    output_path=tasks_path,
                    obj_thresh=args.obj_thresh,
                    iou_thresh=args.iou_thresh,
                    min_class_confidence_thresh=args.min_confidence,
                    max_class_confidence_thresh=args.max_confidence,
                )

        elif args.thumbnail_type == "yogo-incorrect":
            if args.path_to_pth is None:
                raise ValueError(
                    "if `--thumbnail-type yogo-incorrect` is provided, `--path-to-pth` must also be provided"
                )

            def func(image_path: Path, label_path: Path, tasks_path: Path) -> None:
                create_correctness_filtered_tasks_file_from_YOGO(
                    path_to_images=image_path,
                    path_to_labels=label_path,
                    path_to_pth=args.path_to_pth,
                    output_path=tasks_path,
                    obj_thresh=args.obj_thresh,
                )

        else:
            raise NotImplementedError(
                f"somehow got invalid thumbnail type {args.thumbnail_type}"
            )

        if args.path_to_run:
            tasks_information = create_tasks_file_from_path_to_run(
                args.path_to_run,
                tasks_dir / "thumbnail_correction_task_0.json",
                func,
                throw_if_no_labels=(args.thumbnail_type != "yogo-confidence"),
            )
            tasks_and_labels_paths = [tasks_information]
        elif args.path_to_labelled_data_ddf:
            tasks_and_labels_paths = create_tasks_files_from_path_to_labelled_data_ddf(
                args.path_to_labelled_data_ddf, tasks_dir, func
            )
        else:
            raise RuntimeError(
                "one of --path-to-labelled-data-ddf or --path-to-run must be provided"
            )

        create_thumbnails_from_tasks_maps(
            args.path_to_output_dir,
            tasks_and_labels_paths,
            tasks_dir,
            class_dirs,
            classes_to_ignore=args.ignore_class or [],
            image_server_root=args.image_server_relative_parent_override,
        )

        shutil.copy("see_in_context.py", args.path_to_output_dir)

        with open(args.path_to_output_dir / "config_print.txt", "w") as f:
            f.write(repr(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
