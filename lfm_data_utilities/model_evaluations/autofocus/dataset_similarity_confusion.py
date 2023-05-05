#! /usr/bin/env python3

"""
Given a folder of images, calculate the similarity between each pair of
images, and plot a confusion matrix with the results.

Similarity is calculated using cv2.matchTemplate
"""

import os
import cv2
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from functools import partial

from lfm_data_utilities import utils

# sometimes bruno doesn't like me plotting, even if I am just
# saving and not displaying any plots - so here is a magic incantation
# to please Bruno the Great
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def generate_confusion_matrix(folder: Path, output_dir: Path = Path(".")):
    # Get list of images in folder
    images = list(folder.glob("*.png"))
    images.sort()

    # Load images (woa! we can fit them in ram!)
    imgs = []
    for img in images:
        imgs.append(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))

    indices = [(i, j) for i in range(len(imgs)) for j in range(i + 1, len(imgs))]

    # Calculate similarity
    conf = np.zeros((len(imgs), len(imgs)))
    for i in range(len(imgs)):
        conf[i, i] = 1.0

    def update_confusion_matrix(ij):
        i, j = ij
        res = cv2.matchTemplate(imgs[i], imgs[j], cv2.TM_CCOEFF_NORMED)
        conf[i, j] = res[0, 0]
        conf[j, i] = res[0, 0]

    utils.multithread_map_unordered(
        indices,
        update_confusion_matrix,
        verbose=False,
    )

    # directory structure here is
    # ssaf_data_dir/
    #   training_data/
    #       0/
    #       1/
    #       2/
    #       ...
    # we want to save the confusion matricies in ssaf_data_dir
    # and folder is 0/ or 1/ or ...
    folder_name = folder.name  # this is the step

    # Save confusion matrix
    np.save(str(output_dir / f"confusion_matrix_{folder_name}.npy"), conf)

    # and the confusion matrix image
    im = Image.fromarray(255 * conf)
    im = im.convert("L")
    im.save(str(output_dir / f"confusion_matrix_raw_{folder_name}.png"))

    # Plot confusion matrix
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.imshow(conf, cmap="gray")
    plt.colorbar()
    plt.title(f"{folder.name}")
    plt.gca().xaxis.tick_top()
    plt.xlabel("Image")
    plt.ylabel("Image")
    plt.tight_layout()

    plt.savefig(str(output_dir / f"confusion_matrix_raw_{folder_name}.png"), dpi=800)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("-f", "--folder", type=Path, help="path to of images")
    data_source.add_argument(
        "--ssaf-data-dir",
        type=Path,
        help=(
            "path to folder of training data - will find all `training_data` dirs"
            "to find images, and will create a graph per each `training_data` dir"
        ),
    )
    parser.add_argument("-o", "--output", type=Path, help="Output file name")
    args = parser.parse_args()

    if args.folder:
        generate_confusion_matrix(args.folder, args.output)
    else:
        for training_data_dir in args.ssaf_data_dir.rglob("training_data"):
            step_folders = [
                step_folder
                for step_folder in training_data_dir.iterdir()
                if step_folder.is_dir()
            ]
            output = training_data_dir.parent / "confusion_matricies"
            output.mkdir(exist_ok=True)
            utils.multiprocess_fn(
                step_folders,
                partial(generate_confusion_matrix, output_dir=output),
                ordered=False,
                verbose=False,
            )
