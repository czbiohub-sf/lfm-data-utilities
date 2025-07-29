# -*- coding: utf-8 -*-
""" Compile cellpose masks and corresponding np files by dataset
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

DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/cellpose-data/compiled-cellpose-masks')

clindata = pd.read_csv(f"{PTH}/../inputs/rwanda_mch_data.csv")


def compile_masks(d: Path):
    
    expt_id = d.stem
    disk_id = d.parent.parent.stem

    masks = np.zeros((100, 772, 1032))

    for i, f in enumerate(Path(f'{d}/sub_sample_imgs').glob("*.png")):
        with open(f'{DATA_DIR}/../per-img-masks/{disk_id}_{expt_id}{f.stem}.npy', 'rb') as fnp:
            img_mask = np.load(fnp)
            masks[i, :, :] = img_mask
                
    with open(f'{DATA_DIR}/{disk_id}_{expt_id}.npy', 'wb') as fsave:
        np.save(fsave, masks)


print(f'\n***** Processing batch from: {csv} *****\n')
for dataset in tqdm(clindata['path'].to_list(), desc='Dataset'):
    compile_masks(Path(dataset))

