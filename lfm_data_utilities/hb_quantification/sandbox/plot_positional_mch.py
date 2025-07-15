# -*- coding: utf-8 -*-
""" Plot MCH across FOV from subsample images in datasets with clinical MCH
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Takes in a .csv containing:
- Estimated MCH for an individual cell
- XY position of given cell
- Path to image containing given cell

To generate the above .csv, first run get_positional_mch.py to estimate MCH.
Compile all datasets using compile_positional_datasets.py, which will output
the necessary .csv.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import binned_statistic_2d


# DATASET='disk6'
# DATASET='disk7'
DATASET='disk8'

f = '../outputs/rwanda_mch_data_positions_processed.csv'
df = pd.read_csv(f)

# Filter by dataset
dff = df[df['path'].str.contains(DATASET)]
print(dff)

# Bin results
stat, x_edges, y_edges, binnumber = binned_statistic_2d(
    dff['pos_x_estimate'],
    dff['pos_y_estimate'],
    dff['mch_estimate'],
    statistic='mean',
    bins=(30, 24)  # adjust bins as needed
)

# Plot cell count heatmap
counts, _, _ = np.histogram2d(dff['pos_x_estimate'], dff['pos_y_estimate'], bins=(30, 24))
plt.figure()
plt.imshow(counts.T, origin='lower',  cmap='viridis',
    # extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    # aspect='auto', cmap='viridis'
)
plt.colorbar(label='Cells counted')
plt.title(f'Estimated cell count heatmap ({DATASET})')
plt.xticks([])
plt.yticks([])
plt.savefig(f'../plots/cell_count_heatmap_{DATASET}.png')

# Plot MCH heatmap 
plt.figure()
plt.imshow(stat.T, origin='lower',  cmap='viridis',
    # extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    # aspect='auto', cmap='viridis'
)
plt.colorbar(label='Estimated MCH (pg)')
plt.title(f'Estimated MCH heatmap ({DATASET})')
plt.xticks([])
plt.yticks([])
plt.savefig(f'../plots/mch_heatmap_{DATASET}.png')
plt.show()