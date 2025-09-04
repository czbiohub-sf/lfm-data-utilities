# -*- coding: utf-8 -*-
""" Save cellpose masks from subsample images
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Takes in a .csv containing:
- Paths to all datasets
- Corresponding clinical MCH values
Runs cellpose-SAM and saves the masks in a .npy file, each labelled as
<disk id>_<experiment_id>_<img_id>.npy
"""

import time
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cellpose import models, core, io, plot
from datetime import datetime
from pathlib import Path
from tqdm import trange, tqdm
from natsort import natsorted
from typing import Tuple, Optional, List


##### CONSTANTS / CONVERSIONS #####
PTH = Path(__file__).parent


##### SEGMENTATION PARAMETERS #####
flow_threshold = 0.0
cellprob_threshold = -1
tile_norm_blocksize = 0


def init_model() -> models.CellposeModel:
    # io.logger_setup() # run this to get printing of progress
    model = models.CellposeModel(gpu=True)
    # model = models.CellposeModel(gpu=False)

    return model


##### PIPELINE #####
def get_masks(f: Path) -> Tuple[float, float, float]:
    try:
        img = io.imread(f)
        # print(f'Loaded img {f.name}')

        t0 = time.perf_counter()
        masks, flows, styles = model.eval(
            img,
            batch_size=32,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize},
        )

        return masks

    except Exception as e:
        print(f"Could not process {f}:\n{e}")
        pass


def save_masks(f: Path):
    file_id = f.stem
    expt_id = f.parent.parent.stem
    disk_id = f.parent.parent.parent.parent.stem

    masks = get_masks(f)

    with open(f"{DATA_DIR}/{disk_id}_{expt_id}{file_id}.npy", "wb") as npf:
        np.save(npf, masks)


def save_dataset_metadata(dataset: Path) -> Optional[float]:
    try:
        dir = dataset / "sub_sample_imgs"

        if not dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir}")

        # list all files
        files = natsorted(
            [
                f
                for f in dir.glob("*.png")
                if "_masks" not in f.name and "_flows" not in f.name
            ]
        )

        if len(files) == 0:
            raise FileNotFoundError(
                "No image files found, did you specify the correct folder and extension?"
            )
        else:
            print(f"{len(files)} images in directory: {dir}")

        for f in tqdm(files, desc="Image  "):
            save_masks(f)

    except Exception as e:
        print(f"ERROR:\n{e}\n")
        return None


##### RUN SCRIPT #####

# ASCII art descriptor
print("  ___  __    __  __ _  __  ___   __   __      _  _   ___  _  _ ")
print(" / __)(  )  (  )(  ( \\(  )/ __) / _\\ (  )    ( \\/ ) / __)/ )( \\")
print("( (__ / (_/\\ )( /    / )(( (__ /    \\/ (_/\\  / \\/ \\( (__ ) __ (")
print(" \\___)\\____/(__)\\_)__)(__)\\___)\\_/\\_/\\____/  \\_)(_/ \\___)\\_)(_/")

csv = input(
    "Path to .csv with headers ['path', 'mch_pg'] or click Enter to manually run single dataset:\n"
)

if not csv == "":
    DATA_DIR = Path(
        "/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/hb/cellpose-masks"
    )

    df = pd.read_csv(csv)
    if not ("path" in df.columns and "mch_pg" in df.columns):
        raise ValueError(".csv is missing 'path' and/or 'mch_pg' header(s)")

    model = init_model()

    print(f"\n***** Processing batch from: {csv} *****\n")
    for dataset in tqdm(df["path"].to_list(), desc="Dataset"):
        save_dataset_metadata(Path(dataset))
else:
    DATA_DIR = "temp"

    expt = input(
        "Enter single dataset path as per .../LFM_scope/<dataset>/sub_sample_imgs/:\n"
    )
    dataset = Path(f"/hpc/projects/group.bioengineering/LFM_scope/{expt}")

    model = init_model()

    save_dataset_metadata(dataset)
