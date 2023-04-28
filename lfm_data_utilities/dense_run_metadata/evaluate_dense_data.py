#! /usr/bin/env python3

import argparse

from pathlib import Path


""" Evaluation of dense metrics

What qualifies as a good run?
"""


def eval_data_csv(data_path: Path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate dense data from runs")
    parser.add_argument("path_to_dense_data_dir", type=Path, help="path to directory of 'dense data'")
    parser.add_argument("--ssaf-failrate", type=float, default=0.9, help="percent of results (in [0,1]) that must succeed for a pass")
    parser.add_argument("--flowrate-failrate", type=float, default=0.9, help="percent of results (in [0,1]) that must succeed for a pass (default 0.9)")
    parser.add_argument("--ssaf-steprange", type=float, default=2.0, help="step size for a pass - i.e. a value must be in [-val, val] (default 2.0)")

    args = parser.parse_args()

    if not args.path_to_dense_data_dir.exists() or not args.path_to_dense_data_dir.is_dir():
        raise ValueError(f"{args.path_to_dense_data_dir} does not exist or is not a dir")

    for data_csv in args.path_to_dense_data_dir.rglob("data.csv"):
        pass
