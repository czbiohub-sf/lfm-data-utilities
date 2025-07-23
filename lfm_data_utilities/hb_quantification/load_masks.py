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


##### CONSTANTS / CONVERSIONS #####
PTH = Path(__file__).parent


def load_masks(f: Path):
    file_id = f.stem
    expt_id = f.parent.parent.stem
    disk_id = f.parent.parent.parent.parent.stem
    
    with open(f'{DATA_DIR}/{disk_id}_{expt_id}{file_id}.npy', 'rb') as fnp:
        mask = np.load(fnp)

    fstr = str(f)

    return disk_id, expt_id, file_id, mask

##### RUN SCRIPT #####

DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/cellpose-masks')
clindata = pd.read_csv("inputs/rwanda_mch_data.csv")

print(f'\n***** Processing batch from: {csv} *****\n')
res = [load_masks(Path(f)) for dataset in tqdm(clindata['path'].to_list(), desc='Dataset') for f in Path(f'{dataset}/sub_sample_imgs').glob("*.png")]

metadata = pd.DataFrame()
metadata['disk'] = res[:, 0]
metadata['expt'] = res[:, 1]
metadata['file'] = res[:, 2]
metadata['masks'] = res[:, 3]
