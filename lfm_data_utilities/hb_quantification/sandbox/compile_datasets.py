# -*- coding: utf-8 -*-
""" Compile MCH and other metadata from individual dataset .csvs
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Use where multi-dataset compilation at end of get_mch.py fails to save.
Requires:
- Individual .csvs for image processing outputs (include MCH, etc.)
- Compiled .csv for all clinical metadata
"""

import numpy as np
import pandas as pd

from pathlib import Path

df = pd.DataFrame()
files = [f for f in Path("../outputs/rwanda_mch_data").iterdir()]

clinical_df = pd.read_csv("reexport_rwanda_metadata.csv")


def extract_metadata(file: str) -> tuple[float, float, float, str]:
    dff = pd.read_csv(file)
    return (
        np.mean(dff["mch_pg"]),
        np.mean(dff["vol_fl"]),
        np.mean(dff["hct"]),
        np.mean(dff["cell_count"]),
        Path(dff["dir"].loc[0]).parent.parent,
    )


def match_metadata(file: str) -> tuple[str, str, str]:
    file = str(file).replace("\\", "/")
    i = clinical_df.loc[clinical_df["path"] == file].index.values[0]
    row = clinical_df.loc[i]
    return (
        row["mch_pg"],
        row["mcv_fl"],
        row["hct_percent"],
    )


metadata = np.array([extract_metadata(f) for f in files])
df["mch_estimate"] = metadata[:, 0]
df["vol_estimate"] = metadata[:, 1]
df["hct_estimate"] = metadata[:, 2]
df["cell_count_estimate"] = metadata[:, 3]
df["path"] = metadata[:, 4]

clinical_metadata = np.array([match_metadata(f) for f in df["path"]])
df["mch_clinical"] = clinical_metadata[:, 0]
df["vol_clinical"] = clinical_metadata[:, 1]
df["hct_clinical"] = clinical_metadata[:, 2]

df.to_csv("../outputs/rwanda_mch_data_processed.csv")
print("Compiled data and wrote to ../outputs/rwanda_mch_data_processed.csv")
