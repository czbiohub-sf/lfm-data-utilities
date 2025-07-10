# -*- coding: utf-8 -*-
""" Compile MCH and other metadata from individual dataset .csvs
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Use where multi-dataset compilation at end of get_positional_mch.py fails to save.
Requires:
- Individual .csvs for image processing outputs (include MCH, etc.)
"""

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import trange, tqdm


df = pd.DataFrame()
files = [f for f in Path('../outputs/rwanda_mch_data_positions').iterdir()]

def extract_metadata(file: str) -> tuple[float, float, float, str]:
    try:
        dff = pd.read_csv(file)
    except:
        print(f'Failed on dataset {file}')
        raise
    out = np.array(
        [
            dff['mch_pg'],
            dff['pos_x'],
            dff['pos_y'],
            dff['dir'],
        ]
    ).T
    return out

metadata = np.vstack([extract_metadata(f) for f in tqdm(files)])

df['mch_estimate'] = metadata[:, 0]
df['pos_x_estimate'] = metadata[:, 1]
df['pos_y_estimate'] = metadata[:, 2]
df['path'] = metadata[:, 3]

df.to_csv('../outputs/rwanda_mch_data_positions_processed.csv')
print("Compiled data and wrote to ../outputs/rwanda_mch_data_positions_processed.csv")