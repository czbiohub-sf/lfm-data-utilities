# -*- coding: utf-8 -*-
""" Load all cellpose masks
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06
"""

import time
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cellpose import models, core, io, plot
from datetime import datetime
from pathlib import Path
from tqdm import trange, tqdm
from natsort import natsorted
from typing import Tuple, Optional, List


##### CONSTANTS / CONVERSIONS #####
PTH = Path(__file__).parent


def load_masks(f: Path):
    with open(f, 'rb') as fnp:
        mask = np.load(fnp)

    fstr = str(f)
    disk_id = fstr[5]
    expt_id = fstr[6:23]
    file_id = fstr[24:29]

    return disk_id, expt_id, file_id, mask

##### RUN SCRIPT #####

DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/cellpose-masks')
metadata = pd.read_csv("inputs/rwanda_mch_data.csv")

print(f'\n***** Processing batch from: {csv} *****\n')
res = [load_masks(Path(dataset)) for dataset in tqdm(df['path'].to_list(), desc='Dataset')]

metadata = pd.DataFrame()
metadata['disk'] = res[:, 0]
metadata['expt'] = res[:, 1]
metadata['file'] = res[:, 2]
metadata['masks'] = res[:, 3]