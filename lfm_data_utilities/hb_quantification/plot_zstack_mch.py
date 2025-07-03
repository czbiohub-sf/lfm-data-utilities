# -*- coding: utf-8 -*-
""" Plot zstack MCH as a function of position
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Plots data in the .csv output of get_zstack_mch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


csv = input("Path to zstack .csv with headers ['path', 'mch_estimate]:\n")
df = pd.read_csv(csv)

pos = [Path(pth).stem[0:3] for pth in df['path']]
mch = df['mch_estimate']
vol = df['vol_estimate']

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

# plt.plot([15, 50], [15, 50], linestyle='--', c='orange', lw=1)
ax1.scatter(pos, mch)
ax1.set_xlabel('Motor step')
ax1.set_ylabel('Estimated MCH (pg)')
ax1.set_title('Estimated MCH vs z-position')
ax1.tick_params(axis='x', labelrotation=90)

ax2.scatter(pos, vol)
ax2.set_xlabel('Motor step')
ax2.set_ylabel('Estimated volume (fL)')
ax2.set_title('Estimated volume vs z-position')
ax2.tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()