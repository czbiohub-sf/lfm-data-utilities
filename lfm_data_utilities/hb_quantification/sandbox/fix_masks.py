# -*- coding: utf-8 -*-
""" Transpose cellpose masks and save corrected files
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

DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/compiled-cellpose-masks')


def compile_masks(f: Path):
    for i in enumerate(DATA_DIR.glob("*.npy")):
    
    with open(f, 'rb') as fnp:
        mask = np.load(fnp)

    with open(f'{DATA_DIR}/../fixed-cellpose-masks/{f.name}', 'wb') as fnp:
        np.save(fnp, mask.T)

for file in tqdm(DATA_DIR.glob("*.npy"), desc='Dataset'):
    compile_masks(file)
