#! /usr/bin/env python3

""" Jobs for this file:

1 find all run folders (by zarr file)
2 run SSAF, YOGO, and flowrate on all images
3 save results in a csv file
4 save metadata in a yml file
"""

import argparse

from lfm_data_utilities import utils


if __name__ == "__main__":
    # argument parser for run sets, output directory
    parser = argparse.ArgumentParser("create dense data from runs for vetting")
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument("output_dir", type=Path, default=None, help="path to output directory - defaults to path_to_runset/../output")

    args = parser.parse_args()
    path_to_runset = args.path_to_runset

    if not path_to_runset.exists():
        raise ValueError(f"{path_to_runset} does not exist")

    paths = utils.get_all_dataset_paths(args.path_to_runset)
