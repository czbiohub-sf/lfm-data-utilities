# -*- coding: utf-8 -*-
""" Fit estimated MCH against clinical MCH
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Performs a linear fit of estimated MCH to clinical MCH in the .csv output of 
get_mch.py. Plots results and computes residuals for both the raw and corrected data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import least_squares


csv = input("Path to .csv with headers ['path', 'mch_pg', 'mch_estimate]:\n")

# Load and clean up dataframe
df = pd.read_csv(csv)
df = df.dropna()

# Ignore outliers
df = df[df['mch_pg'] < 50]
df = df[df['mch_pg'] > 10]

actual = np.array(df['mch_pg'])
estimate = np.array(df['mch_estimate'])

# Linear function
def scale(c0, x):
    return np.multiply(c0[0], x)

# Function to be minimized for least squares fit
def min(c0):
    return scale(c0, estimate) - actual

# Perform best fit
c0 = [0.9]
res = least_squares(min, c0)
print(res)

corrected = scale(res.x, estimate)


##### Raw MCH estimates #####
plt.figure()
plt.scatter(actual, estimate)
plt.plot([0, 45], [0,45], linestyle=':', color='orange')

plt.title("MCH estimate vs clinical MCH")
plt.xlabel("Clinical MCH (pg)")
plt.ylabel("Raw MCH estimate (pg)")


##### Corrected MCH estimates #####
plt.figure()
plt.scatter(actual, corrected)
plt.plot([0, 45], [0,45], linestyle=':', color='orange')

plt.title("Corrected MCH estimate vs clinical MCH")
plt.xlabel("Clinical MCH (pg)")
plt.ylabel("Corrected MCH estimate (pg)")


##### Raw MCH residuals #####
residuals = actual - estimate
print(f'Residuals mean = {np.mean(residuals)}')

plt.figure()
plt.hist(residuals)
plt.title("Residuals without correction")
plt.xlabel("Residual (pg)")
plt.ylabel("#")


##### Corrected MCH residuals #####
residuals = corrected - estimate
print(f'Residuals mean = {np.mean(residuals)}')

plt.figure()
plt.hist(residuals)
plt.title("Residuals with correction")
plt.xlabel("Residual (pg)")
plt.ylabel("#")

plt.show()