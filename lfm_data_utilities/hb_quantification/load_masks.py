# -*- coding: utf-8 -*-
""" Load all cellpose masks
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06
"""

import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import trange, tqdm
from natsort import natsorted
from typing import Tuple, Optional, List
from scipy.stats import binned_statistic_2d


##### CONSTANTS / CONVERSIONS #####
PTH = Path(__file__).parent

DATASETS= [
    'disk6',
    'disk7',
    'disk8',
]

def calc_pos_and_area(masks: np.ndarray[int], cell_id: int) -> list[int, float, float]:
    px_area = np.sum(masks == cell_id)
    px_row, px_col = np.where(masks == cell_id)
    return px_area, np.mean(px_row), np.mean(px_col)

def load_masks(f: Path):
    file_id = f.stem
    expt_id = f.parent.parent.stem
    disk_id = f.parent.parent.parent.parent.stem
    
    with open(f'{DATA_DIR}/{disk_id}_{expt_id}{file_id}.npy', 'rb') as fnp:
        mask = np.load(fnp)

    return disk_id, expt_id, file_id, mask

##### RUN SCRIPT #####

DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/cellpose-masks')
clindata = pd.read_csv(f"{PTH}/inputs/rwanda_mch_data.csv")

print(f'\n***** Processing batch from: {csv} *****\n')
res = [load_masks(Path(f)) for dataset in tqdm(clindata['path'].to_list(), desc='Dataset') for f in Path(f'{dataset}/sub_sample_imgs').glob("*.png")]

metadata = pd.DataFrame()

metadata['disk'] = [row[0] for row in res]
metadata['expt'] = [row[1] for row in res]
metadata['file'] = [row[2] for row in res]
metadata['masks'] = [row[3] for row in res]

for DATASET in DATASETS:

    dff = metadata[metadata['disk'].str.contains(DATASET)]

    pos_and_area = np.array([calc_pos_and_area(mask, cell_id) for mask in tqdm(dff['masks']) for cell_id in range(np.max(mask))])

    # Bin results
    stat, x_edges, y_edges, binnumber = binned_statistic_2d(
        pos_and_area[:,0],
        pos_and_area[:, 1],
        pos_and_area[:, 2],
        statistic='mean',
        bins=(30, 24)  # adjust bins as needed
    )

    # Plot area heatmap 
    plt.figure()
    plt.imshow(stat.T, origin='lower',  cmap='viridis',
        # extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        # aspect='auto', cmap='viridis'
    )
    plt.colorbar(label='Estimated area (pixels)')
    plt.title(f'Estimated area heatmap ({DATASET})')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'../plots/area_heatmap_{DATASET}.png')
    plt.show()
