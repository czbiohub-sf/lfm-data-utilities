from main import calc_qf
from pathlib import Path
import polars as pl

sets = Path("/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/SingleShotAutofocus/unsorted/")

data = []

for d in sets.iterdir():
    if d.is_dir():
        for dd in d.iterdir():
            if dd.is_dir():
                try:
                    qf = calc_qf(dd)
                    data.append({
                        "run": dd.name,
                        "run_set": d.name,
                        "a": qf.convert().coef[2]
                    })
                except Exception as e:
                    print(f"{dd} threw {e}")


df = pl.from_dicts(data)
df.write_csv("coeffs.csv")

# coding: utf-8
import sys
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # E:

plt.rcParams.update({'font.size': 12})

df = pl.read_csv(sys.argv[1])
pddf = df.groupby("run_set").agg("a").to_pandas().explode('a')
pddf['a'] = pd.to_numeric(pddf['a'])

# Plotting the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='run_set', y='a', data=pddf, order=sorted(pddf['run_set'].unique()))
plt.xticks(rotation=45)
plt.tight_layout()
import time
plt.savefig(f"violin.png_{time.time()}.png")
