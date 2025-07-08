# -*- coding: utf-8 -*-
""" Compare estimated MCH at start vs end of experiment
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


df = pd.DataFrame()
files = [f for f in Path('../outputs/rwanda_mch_data').iterdir()]

clinical_df = pd.read_csv('reexport_rwanda_metadata.csv')

def extract_metadata(file: str) -> tuple[float, float, float, str]:
    dff = pd.read_csv(file)
    return (
        np.mean(dff['mch_pg'].head(10)),
        np.std(dff['mch_pg'].head(10)),
        np.mean(dff['mch_pg'].tail(10)),
        np.std(dff['mch_pg'].tail(10)),
        len(dff),
        # np.mean(dff['vol_fl']),
        # np.mean(dff['hct']),
        Path(dff['dir'].loc[0]).parent.parent,
    )

def match_metadata(file: str) -> tuple[str, str, str]:
    i = clinical_df.loc[clinical_df['path']==str(file)].index.values[0]
    row = clinical_df.loc[i]
    return row['mch_pg']

metadata = np.array([extract_metadata(f) for f in files])
mch_start = metadata[:, 0]
mch_start_std = metadata[:, 1]
mch_end = metadata[:, 2]
mch_end_std = metadata[:, 3]
img_count = metadata[:, 4]
pths = metadata[:, 5]

mch_clinical = np.array([match_metadata(f) for f in pths])

plt.plot([10, 40], [10, 40], color='grey', linestyle='--')

# Plot cases with completed data collection
count_filt = img_count == 100
# plt.scatter(mch_clinical[count_filt], mch_start[count_filt], color='r', s=1, label='First 10 frames')
# plt.scatter(mch_clinical[count_filt], mch_end[count_filt], color='b', s=1, label="Last 10 frames")
plt.scatter(mch_clinical[count_filt], mch_start[count_filt], s=3, label='First 10 frames')
plt.scatter(mch_clinical[count_filt], mch_end[count_filt], s=3, label="Last 10 frames")

plt.title("Estimated MCH for first and last 10 frames")
plt.xlabel("Clinical MCH (pg)")
plt.ylabel("Estimated MCH (pg)")


# Plot cases with incomplete data collection
# plt.scatter(mch_clinical[~count_filt], mch_start[~count_filt], color='r', s=1, marker='+')
# plt.scatter(mch_clinical[~count_filt], mch_end[~count_filt], color='b', s=1, marker='+')

plt.legend()
plt.show()

print(f'Incomplete datasets: {np.sum(~count_filt)}')
print(f"Average diff (end MCH - start MCH) = {np.mean(mch_end-mch_start):.3e} pg")
print(f"Average std (start) = {np.mean(mch_start_std):.3f} pg")
print(f"Average std (end) = {np.mean(mch_end_std):.3f} pg")
# df.to_csv('../outputs/rwanda_mch_data_processed.csv')
# print("Compiled data and wrote to ../outputs/rwanda_mch_data_processed.csv")