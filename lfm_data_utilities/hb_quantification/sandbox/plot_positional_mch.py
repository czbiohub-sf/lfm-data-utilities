# -*- coding: utf-8 -*-
""" Plot estimated Hb metric against clinical Hb metric
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Plots data from the .csv output of get_mch.py
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import binned_statistic_2d


DATASET='disk6'

f = '../outputs/rwanda_mch_data_positions_processed.csv'
df = pd.read_csv(f)

# Filter by dataset
dff = df[df['path'].str.contains(DATASET)]

# Bin results
stat, x_edges, y_edges, binnumber = binned_statistic_2d(
    dff['pos_x_estimate'],
    dff['pos_y_estimate'],
    dff['mch_estimate'],
    statistic='mean',
    bins=30  # adjust bins as needed
)

# Plot cell count heatmap
plt.hist2d(dff['pos_x_estimate'], dff['pos_y_estimate'], bins=30, cmap='viridis')
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