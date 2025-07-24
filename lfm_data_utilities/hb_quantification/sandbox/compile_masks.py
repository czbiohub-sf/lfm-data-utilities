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

clindata = pd.read_csv("inputs/rwanda_mch_data.csv")


def compile_masks(d: Path):
    expt_id = d.stem
    disk_id = d.parent.parent.stem

    concat = np.vstack()

    masks = np.zeros((772, 1032, 100))

    for i in enumerate(Path(f'{d}/sub_sample_imgs').glob("*.png")):
        with open(f'{DATA_DIR}/{disk_id}_{expt_id}{file_id}.npy', 'rb') as fnp:
            masks[:, :, i] = np.load(fnp)

    print(masks)

    # return disk_id, expt_id, file_id, mask

print(f'\n***** Processing batch from: {csv} *****\n')
res = [compile_masks(Path(dataset)) for dataset in tqdm(clindata['path'].to_list()[0], desc='Dataset')]
