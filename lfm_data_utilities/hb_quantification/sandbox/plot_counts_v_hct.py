# -*- coding: utf-8 -*-
""" Plot estimated cell count against clinical hematocrit
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Plots data from the .csv output of get_mch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv = input(f"Path to .csv with headers ['path', 'hct_clinical', 'cell_count_estimate]:\n")
df = pd.read_csv(csv)

# min = np.min(df[f'hct_clinical'])
# max = np.max(df[f'hct_clinical'])
# plt.plot([min, max], [min, max], linestyle='--', c='orange', lw=1)

plt.scatter(df[f'hct_clinical'], df[f'cell_count_estimate'])

plt.xlabel(f'Clinical hematocrit (%)')
plt.ylabel(f'Estimated cell count')
plt.title(f'Estimated cell count vs clinical hematocrit')
plt.show()