#! /usr/bin/env python3

import json

from functools import cache
from pathlib import Path
from dataclasses import dataclass

from yogo.data.dataset_definition_file import DatasetDefinition

"""
messy, but correct, file used to track the thumbnails that we used for training.
These track thumbnails that were EXPLICITLY LOOKED-AT BY A PERSON
"""


@cache
def get_all_root_dirs_used_in_training():
    ddf = DatasetDefinition.from_yaml(
        Path(
            "/home/axel.jacobsen/celldiagnosis/dataset_defs/fine-tuning/all-dataset-subsets.yml"
        )
    )

    return set(d.label_path.parent for d in ddf.all_dataset_paths)


def all_dirs_in(p: str) -> list[Path]:
    # finds non-empty dirs in p
    return [x for x in Path(p).resolve().iterdir() if x.is_dir() and any(x.iterdir())]


def filter_on_substring(dirs: list[str], substring: str) -> list[str]:
    return [d for d in dirs if substring in d]


class_names = [
    "healthy",
    "ring",
    "trophozoite",
    "schizont",
    "gametocyte",
    "wbc",
    "misc",
]


all_class_folder_names = [
    *class_names,
    *[f"corrected_{cn}" for cn in class_names],
]


@dataclass
class ThumbnailsDir:
    root_path: Path
    is_composite: bool  # multiple runs exported to thumbnails together
    id_to_tasks: dict[str, dict[str, str | int]]
    healthy: list[Path]
    ring: list[Path]
    trophozoite: list[Path]
    schizont: list[Path]
    gametocyte: list[Path]
    wbc: list[Path]
    misc: list[Path]

    @property
    def class_dir_paths(self) -> list[Path]:
        return [
            self.healthy,
            self.ring,
            self.trophozoite,
            self.schizont,
            self.gametocyte,
            self.wbc,
            self.misc,
        ]

    @classmethod
    def from_root_path(cls, root_path: str | Path) -> "ThumbnailsDir":
        root_path = Path(root_path).resolve()
        class_dirs = [
            p for p in all_dirs_in(root_path) if p.name in all_class_folder_names
        ]

        def class_folder_filtering(
            class_dirs: list[Path], class_name: str
        ) -> list[Path]:
            """
            expect 'class_name' and f'corrected_{class_name}' to be in class_dirs

            `class_name` should be full of folders only, `corrected_class_name` should
            be full of images only

            For folders in `class_name`, filter on whether the folder is *not* numeric.
            We've been inconsistent on the naming of the folders, but typically it's
            something like `0_aditi_complete` or `3_completed_paul` or whatever.
            """
            filtered_folders = []

            for d in class_dirs:
                if f"corrected_{class_name}" in d.name:
                    filtered_folders.append(d)
                elif class_name in d.name:
                    for dd in d.iterdir():
                        if not dd.name.isnumeric() and dd.is_dir():
                            filtered_folders.append(dd)

            return filtered_folders

        with open(root_path / "id_to_task_path.json", "r") as f:
            id_to_tasks = json.load(f)

        return cls(
            root_path=root_path,
            is_composite=len(id_to_tasks) > 1,
            id_to_tasks=id_to_tasks,
            healthy=class_folder_filtering(class_dirs, "healthy"),
            ring=class_folder_filtering(class_dirs, "ring"),
            trophozoite=class_folder_filtering(class_dirs, "trophozoite"),
            schizont=class_folder_filtering(class_dirs, "schizont"),
            gametocyte=class_folder_filtering(class_dirs, "gametocyte"),
            wbc=class_folder_filtering(class_dirs, "wbc"),
            misc=class_folder_filtering(class_dirs, "misc"),
        )

    def num_images(self) -> int:
        return sum(
            [len(list(xx.glob("*.png"))) for x in self.class_dir_paths for xx in x]
        )

    def __repr__(self) -> str:
        def format_name(dir_path: Path) -> list[str]:
            if "corrected_" in dir_path.name:
                return dir_path.name
            return f"{dir_path.parent.name}/{dir_path.name}"

        cns = {
            "healthy": ", ".join([format_name(x) for x in self.healthy]),
            "ring": ", ".join([format_name(x) for x in self.ring]),
            "trophozoite": ", ".join([format_name(x) for x in self.trophozoite]),
            "schizont": ", ".join([format_name(x) for x in self.schizont]),
            "gametocyte": ", ".join([format_name(x) for x in self.gametocyte]),
            "wbc": ", ".join([format_name(x) for x in self.wbc]),
            "misc": ", ".join([format_name(x) for x in self.misc]),
        }
        s = (
            f"ThumbnailsDir(\n"
            f"    root_path={self.root_path.name}, {self.is_composite=}, "
            f"{self.num_images()=},\n"
        )
        for k, v in cns.items():
            if len(v) > 0:
                s += f"    {k}={v},\n"
        s += ")"
        return s


def thumbnaildir_used_in_training(thumbnail_dir: ThumbnailsDir):
    should_go_in = True
    for task_dict in thumbnail_dir.id_to_tasks.values():
        label_path = Path(task_dict["label_path"])
        if label_path.parent not in get_all_root_dirs_used_in_training():
            should_go_in = False
            print(f"not in training {label_path=} {thumbnail_dir.root_path=}")

    return should_go_in


life_stage_timecourse = ThumbnailsDir.from_root_path(
    "/hpc/projects/group.bioengineering/LFM_scope/thumbnail-corrections/life-stage-timecourse/thumbnails"
)
uganda_subsets = [
    ThumbnailsDir.from_root_path(d)
    for d in all_dirs_in(
        "/hpc/projects/group.bioengineering/LFM_scope/thumbnail-corrections/Uganda-subsets"
    )
]

all_thumbnail_dirs = [
    atd
    for atd in [
        life_stage_timecourse,
        *uganda_subsets,
    ]
    if (atd.num_images() > 0 and thumbnaildir_used_in_training(atd))
]
