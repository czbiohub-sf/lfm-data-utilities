# -*- coding: utf-8 -*-
""" Plot area heatmap from compiled numpy datasets
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06
"""


import numpy as np

from pathlib import Path
from tqdm import tqdm

DATASET = 'disk6'


def load_masks(f: Path):
    with open(f, 'rb') as fnp:
        mask = np.load(fnp)

    return mask
DATA_DIR = Path('/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/compiled-cellpose-masks')

res = np.vstack([load_masks(f) for f in tqdm(DATA_DIR.glob("*.npy")) if DATASET in str(f)])
print(res.shape)


