#! /usr/bin/env python3

""" Jobs for this file:

1 find all run folders (by zarr file)
2 run SSAF, YOGO, and flowrate on all images
3 save metadata in a yml file
4 save results in a csv file
"""

import csv
import types
import argparse
import warnings
import traceback

from pathlib import Path
from itertools import cycle, chain
from typing import Optional, List, Tuple, DefaultDict
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import git
import torch

import autofocus as af

from tqdm import tqdm
from ruamel.yaml import YAML
from collections import defaultdict

from lfm_data_utilities import utils
from lfm_data_utilities.malaria_labelling.labelling_constants import CLASSES
from lfm_data_utilities.image_processing.flowrate_utils import (
    get_all_flowrates_from_experiment,
)


def write_metadata(
    output_dir: Path,
    autofocus_path_to_pth: Path,
    yogo_path_to_pth: Path,
):
    autofocus_package_id = utils.try_get_package_version_identifier(af)

    # write all the above to meta.yml in output_dir
    yaml = YAML()
    meta = {
        "autofocus_package_id": autofocus_package_id,
        "autofocus_path_to_pth": str(autofocus_path_to_pth.absolute()),
    }
    with open(output_dir / "meta.yml", "w") as f:
        yaml.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create dense data from runs for vetting")
    parser.add_argument("dataset_description_file", type=Path, help="path to dataset_description_file")
    parser.add_argument(
        "path_to_autofocus_pth",
        type=Path,
        default=None,
        help="path to autofocus pth file",
    )

    args = parser.parse_args()

    if not args.dataset_description_file.exists():
        raise ValueError(f"{args.dataset_description_file} does not exist")
    elif not args.path_to_autofocus_pth.exists():
        raise ValueError(f"{args.path_to_autofocus_pth} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = af.dataloader.get_dataloader(
         args.dataset_description_file,
         batch_size=32,
         split_fractions_override={'eval': 1.},
    )

    net = af.model.AutoFocus.from_pth(args.path_to_autofocus_pth)
    net.eval()
    net.to(device)
    net = torch.jit.script(net)

    results: DefaultDict[int, list] = defaultdict(list)
    with torch.no_grad():
        for imgs, labels in tqdm(dataloaders['eval']):
            imgs = imgs.to(device, dtype=torch.float, non_blocking=True)
            labels = labels.to(device, dtype=torch.float, non_blocking=True)
            preds = net(imgs).view(-1)

            for i, label in enumerate(labels):
                results[label.item()].append(preds[i].item())


    # plot results
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), color='k', linestyle='--')
    for label, values in results.items():
        # plot candle plots for each label
        npvalues = np.array(values)
        ax.violinplot(npvalues, positions=[label], showmeans=True, showextrema=False)

    ax.set_xlabel('label')
    ax.set_ylabel('autofocus output')
    # save fig with max definition
    fig.savefig('autofocus_output.png', dpi=1000)
