""" NOTICE

This is modified from https://github.com/heartexlabs/label-studio-converter

Read Apache-2.0 license here https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import sys
import json  # better to use "imports ujson as json" for the best performance
import argparse
from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls
import uuid
import logging

from PIL import Image
from pathlib import Path

from typing import Optional, Tuple

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.WARNING)
logger.addHandler(handler)


class ExpandFullPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def convert_yolo_to_ls(
    input_dir,
    out_file,
    to_name="image",
    from_name="label",
    label_dir_name="labels",
    images_dir_path: Optional[Path] = None,
    out_type="annotations",
    image_root_url="/data/local-files/?d=",
    image_ext=".jpg,.jpeg,.png",
    image_dims: Optional[Tuple[int, int]] = None,
    ignore_images_without_labels: bool = False,
):
    """Convert YOLO labeling to Label Studio JSON
    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param label_dir_name: name of the label dir - e.g. "labels", "yogo-labels"
    :param images_dir_path: path to images dir
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on.
    """

    tasks = []
    logger.info("Reading YOLO notes and categories from %s", input_dir)

    # build categories=>labels dict
    notes_file = Path(input_dir) / "notes.json"
    if notes_file.exists():
        with open(notes_file, "r") as f:
            notes_data = json.load(f)
        categories = {int(v["id"]): v["name"] for v in notes_data["categories"]}
    else:
        classes_file = os.path.join(input_dir, "classes.txt")
        try:
            with open(classes_file) as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            lines = [
                'healthy',
                'ring',
                'trophozoite',
                'schizont',
                'gametocyte',
                'wbc',
                'misc'
            ]

        categories = {i: line for i, line in enumerate(lines)}

    logger.info(f"Found {len(categories)} categories")

    # define directories
    labels_dir = os.path.join(input_dir, label_dir_name)
    if images_dir_path is None:
        images_dir = os.path.join(input_dir, "images")
    else:
        images_dir = str(images_dir_path)

    logger.info("Converting labels from %s", labels_dir)

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    logger.info(f"image extensions->, {image_ext}")

    # loop through images
    for f in os.listdir(images_dir):
        image_file_found_flag = False
        for ext in image_ext:
            if f.endswith(ext):
                image_file = f
                image_file_base = f[0 : -len(ext)]
                image_file_found_flag = True
                break

        if not image_file_found_flag:
            logger.warning(f"missing image file for label {f}")
            continue

        image_root_url += "" if image_root_url.endswith("/") else "/"
        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": image_root_url
                + str(pathname2url(image_file))
            }
        }

        # define coresponding label file and check existence
        label_file = os.path.join(labels_dir, image_file_base + ".txt")

        if os.path.exists(label_file):
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if image_dims is None:
                with Image.open(os.path.join(images_dir, image_file)) as im:
                    image_width, image_height = im.size
            else:
                image_width, image_height = image_dims

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                for line in lines:
                    label_id, x, y, width, height = line.split()
                    x, y, width, height = (
                        float(x),
                        float(y),
                        float(width),
                        float(height),
                    )
                    item = {
                        "id": uuid.uuid4().hex[0:10],
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x - width / 2) * 100,
                            "y": (y - height / 2) * 100,
                            "width": width * 100,
                            "height": height * 100,
                            "rotation": 0,
                            "rectanglelabels": [categories[int(label_id)]],
                        },
                        "to_name": to_name,
                        "from_name": from_name,
                        "image_rotation": 0,
                        "original_width": image_width,
                        "original_height": image_height,
                    }
                    task[out_type][0]["result"].append(item)
        elif ignore_images_without_labels:
            # we don't append task to the total list of tasks
            continue

        tasks.append(task)

    if len(tasks) > 0:
        logger.info("Saving Label Studio JSON to %s", out_file)
        with open(out_file, "w") as out:
            json.dump(tasks, out)
    else:
        logger.error("No labels converted")


def add_parser(subparsers):
    yolo = subparsers.add_parser("yolo")

    yolo.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="directory with YOLO where images, labels, notes.json are located",
        action=ExpandFullPath,
    )
    yolo.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output file with Label Studio JSON tasks",
        default="output.json",
        action=ExpandFullPath,
    )
    yolo.add_argument(
        "--to-name",
        dest="to_name",
        help="object name from Label Studio labeling config",
        default="image",
    )
    yolo.add_argument(
        "--from-name",
        dest="from_name",
        help="control tag name from Label Studio labeling config",
        default="label",
    )
    yolo.add_argument(
        "--out-type",
        dest="out_type",
        help='annotation type - "annotations" or "predictions"',
        default="annotations",
    )
    yolo.add_argument(
        "--image-root-url",
        dest="image_root_url",
        help="root URL path where images will be hosted, e.g.: http://example.com/images",
        default="/data/local-files/?d=",
    )
    yolo.add_argument(
        "--image-ext",
        dest="image_ext",
        help="image extension to search: .jpeg or .jpg, .png",
        default=".jpg",
    )
