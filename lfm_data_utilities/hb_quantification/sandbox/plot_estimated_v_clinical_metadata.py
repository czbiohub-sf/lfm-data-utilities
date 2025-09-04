# -*- coding: utf-8 -*-
""" Plot estimated Hb metric against clinical Hb metric
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Plots data from the .csv output of get_mch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


metric = input("Hemoglobin metric to plot (any of 'hct', 'mch', 'mcv'):\n")
unit = input("Hemoglobin metric unit:\n")
id = metric.upper()
if metric == "mcv":
    metric = "vol"

csv = input(
    f"Path to .csv with headers ['path', '{metric}_clinical', '{metric}_estimate]:\n"
)
df = pd.read_csv(csv)

min = np.min(df[f"{metric}_clinical"])
max = np.max(df[f"{metric}_clinical"])

plt.plot([min, max], [min, max], linestyle="--", c="orange", lw=1)
if "hct":
    plt.scatter(df[f"{metric}_clinical"], np.multiply(100, df[f"{metric}_estimate"]))
else:
    plt.scatter(df[f"{metric}_clinical"], df[f"{metric}_estimate"])

plt.xlabel(f"Clinical {id} ({unit})")
plt.ylabel(f"Estimated {id} ({unit})")
plt.title(f"Image processing estimates vs clinical {id}")
plt.show()
