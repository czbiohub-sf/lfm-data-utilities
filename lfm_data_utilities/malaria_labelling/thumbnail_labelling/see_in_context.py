#! /usr/bin/env python3

import sys
import json

from pathlib import Path
from typing import Dict, Tuple, Union

from PIL import Image


""" This script takes a thumbnail filename as an argument and opens the corresponding image in LFM_scope
"""


def parse_thumbnail_name(thumbnail_name: str) -> Tuple[str, ...]:
    """
    parses a thumbnail name into class, cell_id, and task.json id

    We can just remove the '.png' and split on '_', since class, cell_id, and task_json_id don't
    have underscores in them.
    """
    return tuple(s.strip() for s in thumbnail_name.replace(".png", "").split("_"))


def parse_id_to_task_path(
    id_to_task_path: Union[Path, str]
) -> Dict[str, Dict[str, str]]:
    with open(id_to_task_path, "r") as f:
        return json.load(f)


def search_for_dir_in_parents(p: Path, d: str) -> Path:
    if p.parent == p:
        raise ValueError(f"{d} not found in path")
    return p if p.name == d else search_for_dir_in_parents(p.parent, d)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: see_in_context.py <thumbnail filename>")
        sys.exit(1)

    thumbnail_dir = Path(__file__).parent.absolute()
    thumbnail_name = Path(sys.argv[1]).name

    id_to_task_path = parse_id_to_task_path(thumbnail_dir / "id_to_task_path.json")

    class_name, cell_id, task_json_id = parse_thumbnail_name(thumbnail_name)

    task = thumbnail_dir / "tasks" / id_to_task_path[task_json_id]["task_name"]

    LFM_scope_path = search_for_dir_in_parents(thumbnail_dir, "LFM_scope")

    with open(task) as f:
        ls = json.load(f)
        for task in ls:
            for cell in task["predictions"][0]["result"]:  # type: ignore
                if cell["id"] == cell_id:
                    image_url = task["data"]["image"]  # type: ignore
                    image_path = LFM_scope_path / image_url.replace(
                        "http://localhost:8081/", ""
                    )
                    Image.open(image_path).show()  # type: ignore
                    break
