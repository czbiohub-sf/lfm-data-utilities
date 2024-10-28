"""
Pass in a txt file with a list of exp metadata paths.

Generate the txt file using `find`:
`find /path/to/experiment -name "*exp*.csv" -not -path "*/\*." > metadata_paths.txt`

Each metadata file has a single row with multiple columns.
This script will go collate all those files into a single dataframe where each row
corresponds to a metadata file's contents. The path to the metadata file and its path stem 
will also be included.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collate metadata files into a single dataframe"
    )
    parser.add_argument(
        "metadata_paths", type=str, help="Path to the txt file with metadata paths"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the collated metadata dataframe"
    )
    args = parser.parse_args()

    metadata_paths = []
    with open(args.metadata_paths, "r") as f:
        for line in f:
            metadata_paths.append(line.strip())

    data = []
    for path in tqdm(metadata_paths):
        df = pd.read_csv(path)
        df["metadata_path"] = path
        df["metadata_path_stem"] = Path(path).parent.stem
        data.append(df)

    output_file = Path(Path(args.output_path) / "collated.csv")
    df = pd.concat(data, axis=0, ignore_index=True)
    df.to_csv(output_file, index=False)
    print(f"Saved collated metadata to {output_file}")
