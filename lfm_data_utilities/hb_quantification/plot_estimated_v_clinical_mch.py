# -*- coding: utf-8 -*-
""" Plot estimated MCH against clinical MCH
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Plots data from the .csv output of get_mch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Zoomed in view bounds
MIN = 15
MAX = 45


# csv = input("Path to .csv with headers ['path', 'mch_clinical', 'mch_estimate]:\n")
csv = "outputs/rwanda_mch_data_processed.csv"

DATASET = input(
    "String to filter for in experiment path (eg. 'disk6', 'disk7', 'disk8')\n"
)

# Filter datasets
df = pd.read_csv(csv)
if DATASET:
    dff = df[df["path"].str.contains(DATASET)]
else:
    dff = df

# Error
mse = np.sqrt(np.mean(np.power(dff["mch_clinical"] - dff["mch_estimate"], 2)))
print(f"Mean squared err = {mse}")

# Filter out outliers
min_met = np.logical_and(dff["mch_clinical"] > MIN, dff["mch_estimate"] > MIN)
max_met = np.logical_and(dff["mch_clinical"] < MAX, dff["mch_estimate"] < MAX)
filt = np.logical_and(min_met, max_met)

mse_zoom = np.sqrt(
    np.mean(np.power(dff["mch_clinical"][filt] - dff["mch_estimate"][filt], 2))
)
print(f"Mean squared err (zoomed in)= {mse_zoom}")

# Plot
plt.plot([MIN, MAX], [MIN, MAX], linestyle="--", c="orange", lw=1)
plt.scatter(dff["mch_clinical"], dff["mch_estimate"], s=4)

plt.xlabel("Clinical MCH (pg)")
plt.ylabel("Estimated MCH (pg)")
if DATASET:
    plt.title(f"Image processing estimates vs clinical MCH ({DATASET} only)")
else:
    plt.title("Image processing estimates vs clinical MCH")

# plt.show()
if DATASET:
    plt.savefig(f"plots/mch_{DATASET}_MSE{int(mse)}.png")
else:
    plt.savefig(f"plots/mch_MSE{int(mse)}.png")

# Zoom in on data of interest
plt.xlim((MIN, MAX))
plt.ylim((MIN, MAX))
if DATASET:
    plt.savefig(f"plots/mch_zoom_{DATASET}_MSE{int(mse_zoom)}.png")
else:
    plt.savefig(f"plots/mch_zoom_MSE{int(mse_zoom)}.png")
plt.show()
