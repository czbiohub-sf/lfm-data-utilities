# -*- coding: utf-8 -*-
""" Compute and correlate metadata from pre-generated Cellpose masks
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Takes in a folder of .npy files with Cellpose mask arrays

Outputs a .csv correlating:
- Experiment folder with raw images
- .npy file with Cellpose mask
- Frame ID
- Per cell estimated MCH value
- Per cell segmented area
- Per cell x and y position
"""

import time
import os
import argparse
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from tqdm import trange, tqdm
from natsort import natsorted
from typing import Tuple, Optional, List


##### CONSTANTS / CONVERSIONS #####
DATA_DIR = Path(
    "/hpc/projects/group.bioengineering/LFM_scope/hb_investigations/cellpose-data/compiled-cellpose-masks"
)
PTH = Path(__file__).parent

clindata = pd.read_csv(f"{PTH}/inputs/rwanda_mch_data.csv")

# From https://omlc.org/spectra/hemoglobin/summary.html (evaluated at 406nm)
# molar extinction coefficient: 270548 L / (cm * mol) = 270548e3 cm^2 / mol
# Hb has molar mass 64500e12 pg/mol

EPSILON = 270548e3 / 64500e12  # cm^2 / pg

# 2x binning, 30x mag
LEN_PER_PX = 3.45e-4 * 2 / 40  # cm
AREA_PER_PX = LEN_PER_PX**2  # cm^2


def get_npy_filename(f: str):
    f = Path(f)

    expt_id = f.stem
    disk_id = f.parent.parent.stem

    return f"{DATA_DIR}/{disk_id}_{expt_id}.npy"


def get_mask(f: str):
    return np.load(fnp).astype(int)


def calc_pos_and_area(mask: np.ndarray[int], cell_id: int) -> list[int, float, float]:
    hotspots = mask == cell_id
    px_area = np.sum(hotspots)
    px_row, px_col = np.where(hotspots)
    return px_area, np.mean(px_row), np.mean(px_col)


def calc_mch_pg(pg_img: np.ndarray[float], mask: np.ndarray[int]) -> List[float]:
    for cell_id in range(1, int(np.max(mask))):
        print(mask == cell_id, flush=True)
        print(np.sum(pg_img[mask == cell_id]), flush=True)
    # return [np.sum(pg_img[mask == cell_id]) for cell_id in range(1, int(np.max(mask)))]


def calc_pg_per_px(absorbance: np.ndarray[float]) -> np.ndarray[float]:
    hb_mass = np.multiply(absorbance, AREA_PER_PX / EPSILON)  # pg
    return hb_mass


def get_img_metadata(f: str, mask: np.ndarray) -> Tuple[float, float, float]:
    try:
        img = cv2.imread(f)
        ##
        filt = mask > 0
        BKG = np.mean(img[~filt])
        # print(f'NUM_CELLS = {NUM_CELLS}\nBKG = {BKG}\n')
        ##
        absorbance_img = np.log10(BKG / img)
        pg_img = calc_pg_per_px(absorbance_img)
        # print(pg_img, flush=True)
        ##
        mch = np.array(calc_mch_pg(pg_img, mask))
        pos_and_area = np.array(
            [
                calc_pos_and_area(mask, cell_id)
                for cell_id in range(1, int(np.max(mask)))
            ]
        )
        ##
        print(mch, flush=True)
        ##
        return np.hstack((mch, pos_and_area))
        ##
    except Exception as e:
        print(f"Could not process {f}:\n{e}")
        pass


npy_files = [get_npy_filename(dataset) for dataset in tqdm(clindata["path"].to_list())]
grouped_img_files = [
    natsorted(
        [
            f
            for f in Path(f"{dir}/sub_sample_imgs").glob("*.png")
            if "_masks" not in f.name and "_flows" not in f.name
        ]
    )
    for dir in tqdm(clindata["path"].to_list())
]
img_files = np.array([f for expt_files in grouped_img_files for f in expt_files])
masks = np.vstack(np.array([get_mask(f) for f in tqdm(npy_files[0:10])]))

out = [get_img_metadata(f, masks[i, :, :].T) for i, f in enumerate(img_files[0:10])]
# res = [load_masks(Path(f)) for dataset in tqdm(clindata['path'].to_list(), desc='Dataset') for f in Path(f'{dataset}/sub_sample_imgs').glob("*.png")]
