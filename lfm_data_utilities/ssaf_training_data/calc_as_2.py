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
plt.show()
