# -*- coding: utf-8 -*-
""" Estimate MCH from subsample images in datasets with clinical MCH
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.06

Takes in a .csv containing:
- Paths to all datasets
- Corresponding clinical MCH values
Runs cellpose-SAM and MCH quantification pipeline (similar to cellpose_sandbox.ipynb
workflow) and outputs a new .csv with:
- Paths to all datasets
- Corresponding clinical MCH values
- Estimated MCH values from image processing pipeline
- Estimated MCV
- Estimated Hb 

To plot results, use plot_estimated_v_clinical_mch.py or fit_estimated_v_clinical_mch.py
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

# From https://omlc.org/spectra/hemoglobin/summary.html (evaluated at 406nm)
# molar extinction coefficient: 270548 L / (cm * mol) = 270548e3 cm^2 / mol
# Hb has molar mass 64500e12 pg/mol

EPSILON = 270548e3 / 64500e12 # cm^2 / pg

# 2x binning, 30x mag
LEN_PER_PX = 3.45e-4 * 2/ 40 # cm
AREA_PER_PX = LEN_PER_PX ** 2 # cm^2

# print(f'\nLEN_PER_PX = {LEN_PER_PX} cm\nAREA_PER_PX = {AREA_PER_PX:.3e} cm^2 \nEPSILON = {EPSILON:.3e}')

def calc_position(masks: np.ndarray[int], cell_id: int) -> list[float, float]:
    px_row, px_col = np.where(masks == cell_id)
    return np.mean(px_row), np.mean(px_col)

def calc_pg_per_px(absorbance: np.ndarray[float]) -> np.ndarray[float]:
    hb_mass = np.multiply(absorbance, AREA_PER_PX  / EPSILON) # pg
    return hb_mass

def calc_mch_pg(pg_img: np.ndarray[float], masks: np.ndarray[int]) -> list[float]:
    mch = np.array([np.sum(pg_img[masks == cell_id]) for cell_id in range(np.max(masks))])
    pos = np.array([calc_position(masks, cell_id) for cell_id in range(np.max(masks))])
        
    return np.column_stack((mch, pos))

def calc_hct(masks: np.ndarray[int]) -> float:
    cell_pxs = np.sum(masks == 0)
    bkg_pxs = np.sum(masks != 0)
    return cell_pxs / (cell_pxs + bkg_pxs)

def calc_vol_fl(masks: np.ndarray[int]) -> list[float]:
    return [np.sum(masks == i) * AREA_PER_PX * 5 / 1e-8 for i in range(np.max(masks))] # um^3 = fL


##### SEGMENTATION PARAMETERS #####
flow_threshold = 0.0
cellprob_threshold = -1
tile_norm_blocksize = 0

def init_model() -> models.CellposeModel:
    # io.logger_setup() # run this to get printing of progress
    model = models.CellposeModel(gpu=True)
    return model


##### PIPELINE #####
def get_img_metadata(f: str) -> Tuple[float, float, float]:
    try:
        img = io.imread(f)
        # print(f'Loaded img {f.name}')

        t0 = time.perf_counter()
        masks, flows, styles = model.eval(img, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                        normalize={"tile_norm_blocksize": tile_norm_blocksize})
        t1 = time.perf_counter()
        # print(f'Generated masks for img {f.name} in {t1-t0:.3f}s')

        filt = masks > 0
        NUM_CELLS = np.max(masks)
        BKG = np.mean(img[~filt])
        absorbance_img = np.log10(BKG / img)
        pg_img = calc_pg_per_px(absorbance_img)

        files = [f] * NUM_CELLS

        return np.column_stack((calc_mch_pg(pg_img, masks), files))
    
    except Exception as e:
        print(f'Could not process {f}:\n{e}')
        pass

def get_dataset_metadata(dataset: Path, savedir: Path = Path('data/')) -> Optional[float]:
    try:
        dir = dataset / "sub_sample_imgs"

        if not dir.exists():
            raise FileNotFoundError(f'Directory does not exist: {dir}')

        # list all files
        files = natsorted([f for f in dir.glob("*.png") if "_masks" not in f.name and "_flows" not in f.name])

        if(len(files)==0):
            raise FileNotFoundError("No image files found, did you specify the correct folder and extension?")
        else:
            print(f"{len(files)} images in directory: {dir}")

        mch_out = np.array([row for f in tqdm(files, desc='Image  ') for row in get_img_metadata(f)])

        df = pd.DataFrame()
        df['mch_pg'] = mch_out[:, 0]
        df['pos_x'] = mch_out[:, 1]
        df['pos_y'] = mch_out[:, 2]
        df['dir'] = mch_out[:, 4]

        rn = datetime.now()
        df.to_csv(
            savedir / f'{Path(dataset).stem}positional_hbquant.csv',
            # f'cellpose-hb-data/{Path(dataset).stem}{rn.strftime("%Y%m%d-%H%M%S")}.csv
        )

    except Exception as e:
        print(f'ERROR:\n{e}\n')
        return None


##### RUN SCRIPT #####

# ASCII art descriptor
print("  ___  __    __  __ _  __  ___   __   __      _  _   ___  _  _ ")
print(" / __)(  )  (  )(  ( \\(  )/ __) / _\\ (  )    ( \\/ ) / __)/ )( \\")
print("( (__ / (_/\\ )( /    / )(( (__ /    \\/ (_/\\  / \\/ \\( (__ ) __ (")
print(" \\___)\\____/(__)\\_)__)(__)\\___)\\_/\\_/\\____/  \\_)(_/ \\___)\\_)(_/")

csv = input("Path to .csv with headers ['path', 'mch_pg'] or click Enter to manually run single dataset:\n")

if not csv == '':
    df = pd.read_csv(csv)
    if not ('path' in df.columns and 'mch_pg' in df.columns):
        raise ValueError(".csv is missing 'path' and/or 'mch_pg' header(s)")

    try:
        dataset_name = f'{Path(csv).stem}_positions'
        savedir = PTH / "outputs" / dataset_name
        os.mkdir(savedir)
        # print(f'\nDirectory {savedir} created successfully')
    except FileExistsError:
        # print(f'\nDirectory {savedir} already exists')
        pass

    model = init_model()

    print(f'\n***** Processing batch from: {csv} *****\n')
    for dataset in tqdm(df['path'].to_list(), desc='Dataset'):
        get_dataset_metadata(Path(dataset), savedir=savedir) 

else:
    expt = input("Enter single dataset path as per .../LFM_scope/<dataset>/sub_sample_imgs/:\n")
    dataset = Path(f'/hpc/projects/group.bioengineering/LFM_scope/{expt}')

    model = init_model()

    get_dataset_metadata(dataset)

