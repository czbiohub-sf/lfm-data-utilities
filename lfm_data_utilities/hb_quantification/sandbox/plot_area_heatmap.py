import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import trange, tqdm
from natsort import natsorted
from typing import Tuple, Optional, List
from scipy.stats import binned_statistic_2d


DATA_DIR = Path(
    "/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/compiled-cellpose-masks"
)

DATASETS = [
    "disk6",
    "disk7",
    "disk8",
]


def calc_pos_and_area(masks: np.ndarray[int], cell_id: int) -> list[int, float, float]:
    hotspots = masks == cell_id
    px_area = np.sum(hotspots)
    px_row, px_col = np.where(hotspots)
    return px_area, np.mean(px_row), np.mean(px_col)


def load_mask(f: Path):
    with open(f, "rb") as fnp:
        return np.load(fnp)


for DATASET in DATASETS:
    files = [
        file
        for file in tqdm(DATA_DIR.glob("*.npy"), desc="Dataset")
        if DATASET in str(file)
    ]
    masks = np.vstack(np.array([load_mask(file) for file in tqdm(files)]))

    pos_and_area = np.array(
        [
            calc_pos_and_area(mask, cell_id)
            for mask in tqdm(masks)
            for cell_id in range(int(np.max(mask)))
        ]
    )

    # Bin results
    stat, x_edges, y_edges, binnumber = binned_statistic_2d(
        pos_and_area[:, 1],
        pos_and_area[:, 2],
        pos_and_area[:, 0],
        statistic="mean",
        bins=(30, 24),  # adjust bins as needed
    )

    # Plot area heatmap
    plt.figure()
    plt.imshow(
        stat.T,
        origin="lower",
        cmap="viridis",
        # extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        # aspect='auto', cmap='viridis'
    )
    plt.colorbar(label="Estimated area (pixels)")
    plt.title(f"Estimated area heatmap ({DATASET})")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"../plots/area_heatmap_{DATASET}.png")
    plt.show()
