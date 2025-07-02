# -*- coding: utf-8 -*-
""" Plot estimated MCH against clinical MCH
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Plots data in .csv output by get_mch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


csv = input("Path to .csv with headers ['path', 'mch_pg', 'mch_estimate]:\n")
df = pd.read_csv(csv)

plt.plot([15, 50], [15, 50], linestyle='--', c='orange', lw=1)
plt.scatter(df['mch_pg'], df['mch_estimate'])

plt.xlabel('Clinical MCH (pg)')
plt.ylabel('Estimated MCH (pg)')
plt.title('Image processing estimates vs clinical MCH')
plt.show()