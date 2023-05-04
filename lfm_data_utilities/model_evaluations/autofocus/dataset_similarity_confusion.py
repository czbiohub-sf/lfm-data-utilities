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

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from lfm_data_utilities import utils

# sometimes bruno doesn't like me plotting, even if I am just
# saving and not displaying any plots - so here is a magic incantation
# to please Bruno the Great
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def generate_confusion_matrix(folder: Path, output: Optional[Path] = None):
    # Get list of images in folder
    images = list(folder.glob("*.png"))
    images.sort()

    # Load images (woa! we can fit them in ram!)
    with utils.timing_context_manager(f"loading {len(images)} images!"):
        imgs = []
        for img in images:
            imgs.append(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))

    indices = [(i,j) for i in range(len(imgs)) for j in range(i+1, len(imgs))]

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
        verbose=True,
   )

    # Plot confusion matrix
    plt.rcParams["figure.figsize"] = (12,12)
    plt.imshow(conf, cmap="gray")
    plt.colorbar()
    plt.title(f"{folder.parent.parent.name}")
    plt.gca().xaxis.tick_top()
    plt.xlabel("Image")
    plt.ylabel("Image")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output.with_suffix(".png"), dpi=800)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder containing images")
    parser.add_argument("-o", "--output", type=Path, help="Output file name")
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        raise ValueError(f"Folder {folder} does not exist")

    generate_confusion_matrix(folder, args.output)
