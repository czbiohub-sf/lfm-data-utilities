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
                except Exception as e:
                    print(f"{dd} threw {e}")
                else:
                    data.append({
                        "run": dd.name,
                        "run_set": d.name,
                        "a": qf.convert().coef[2]
                    })

df = pl.from_dicts(data)
print(df)
df.write_csv("coeffs.csv")
