# -*- coding: utf-8 -*-
""" Compute and correlate metadata from pre-generated Cellpose masks
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Takes in a folder of .npy files with Cellpose mask arrays

Outputs a .csv correlating:
- Experiment folder with raw images
- .npy file with Cellpose mask
- Frame ID
- Per cell estimated MCH value
- Per cell segmented area
- Per cell x and y position
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
DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/cellpose-data/per-exp-masks')
PTH = Path(__file__).parent

clindata = pd.read_csv(f"{PTH}/inputs/rwanda_mch_data.csv")

def get_npy_filename(f: str):
    f = Path(f)

    expt_id = f.stem
    disk_id = f.parent.parent.stem

    return f'{DATA_DIR}/{disk_id}_{expt_id}.npy'

npy_files = [get_npy_filename(dataset) for dataset in tqdm(clindata['path'].to_list())]

# res = [load_masks(Path(f)) for dataset in tqdm(clindata['path'].to_list(), desc='Dataset') for f in Path(f'{dataset}/sub_sample_imgs').glob("*.png")]
