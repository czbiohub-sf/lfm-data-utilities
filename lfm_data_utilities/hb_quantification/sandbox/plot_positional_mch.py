# -*- coding: utf-8 -*-
""" Plot estimated Hb metric against clinical Hb metric
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Plots data from the .csv output of get_mch.py
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATASET='disk8'

f = '../outputs/rwanda_mch_data_positions_processed.csv'
df = pd.read_csv(f)

dff = df[df['path'].str.contains(DATASET)]

plt.hist2d(dff['pos_x_estimate'], dff['pos_y_estimate'], bins=30, weights=dff['mch_estimate'], cmap='viridis')
# plt.colorbar(label='')
plt.title(f'Estimated MCH heatmap ({DATASET})')
plt.show()