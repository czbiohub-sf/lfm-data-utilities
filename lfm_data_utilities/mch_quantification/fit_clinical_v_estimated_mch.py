import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import least_squares


csv = input("Path to .csv with headers ['path', 'mch_pg', 'mch_estimate]:\n")

df = pd.read_csv(csv)
df = df.dropna()

df = df[df['mch_pg'] < 50]
df = df[df['mch_pg'] > 10]

###

actual = np.array(df['mch_pg'])
estimate = np.array(df['mch_estimate'])

def scale(c0, x):
    return np.multiply(c0[0], x)

def min(c0):
    return scale(c0, estimate) - actual

c0 = [0.9]
res = least_squares(min, c0)
print(res)

###

plt.figure()
plt.scatter(actual, scale(res.x, estimate))
plt.plot([0, 45], [0,45], linestyle=':', color='orange')

plt.title("Corrected MCH estimate vs clinical MCH")
plt.xlabel("Clinical MCH (pg)")
plt.ylabel("Corrected MCH estimate (pg)")

###

plt.figure()
plt.scatter(actual, estimate)
plt.plot([0, 45], [0,45], linestyle=':', color='orange')

plt.title("MCH estimate vs clinical MCH")
plt.xlabel("Clinical MCH (pg)")
plt.ylabel("Raw MCH estimate (pg)")

###

residuals = actual - estimate
print(f'Residuals mean = {np.mean(residuals)}')

plt.figure()
plt.hist(residuals)
plt.title("Residuals without pixel correction")
plt.xlabel("Residual (pg)")
plt.ylabel("#")

plt.show()