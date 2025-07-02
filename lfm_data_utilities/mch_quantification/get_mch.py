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

To plot results, use plot_estimated_v_clinical_mch.py
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


##### CONSTANTS / CONVERSIONS #####
# From https://omlc.org/spectra/hemoglobin/summary.html (evaluated at 406nm)
# molar extinction coefficient: 270548 L / (cm * mol) = 270548e3 cm^2 / mol
# Hb has molar mass 64500e12 pg/mol

EPSILON = 270548e3 / 64500e12 # cm^2 / pg

# 2x binning, 30x mag
LEN_PER_PX = 3.45e-4 * 2/ 40 # cm
AREA_PER_PX = LEN_PER_PX ** 2 # cm^2

# print(f'\nLEN_PER_PX = {LEN_PER_PX} cm\nAREA_PER_PX = {AREA_PER_PX:.3e} cm^2 \nEPSILON = {EPSILON:.3e}')

def calc_pg_per_px(absorbance):
    hb_mass = np.multiply(absorbance, AREA_PER_PX  / EPSILON) # pg
    return hb_mass


##### SEGMENTATION PARAMETERS #####
flow_threshold = 0.0
cellprob_threshold = -1
tile_norm_blocksize = 0

def init_model():
    # io.logger_setup() # run this to get printing of progress
    model = models.CellposeModel(gpu=True)
    return model


##### PIPELINE #####
def get_img_hb(f):
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
    # print(f'NUM_CELLS = {NUM_CELLS}\nBKG = {BKG}\n')

    absorbance_img = np.log10(BKG / img)
    pg_img = calc_pg_per_px(absorbance_img)

    mchs = [np.sum(pg_img[masks == cell_id]) for cell_id in range(NUM_CELLS)]
    avg_mch = np.mean(mchs)
    # print(f'Per image MCH  = {avg_mch:.3f} pg')

    return avg_mch, NUM_CELLS, BKG 

def get_dataset_hb(dataset: Path, savedir: Path = Path('data/')):
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

        mch_out = np.array([get_img_hb(f) for f in tqdm(files, desc='Image  ')])

        df = pd.DataFrame()
        df['mch_pg'] = mch_out[:, 0]
        df['cell_count'] = mch_out[:, 1]
        df['bkg'] = mch_out[:, 2]
        df['dir'] = files

        rn = datetime.now()
        df.to_csv(
            savedir / f'{Path(dataset).stem}hbquant.csv',
            # f'cellpose-hb-data/{Path(dataset).stem}{rn.strftime("%Y%m%d-%H%M%S")}.csv
        )

        mch = np.mean(mch_out[:, 0])

        print(f'MCH = {mch:.3f} pg')
        print(f'STD of background = {np.std(mch_out[:, 2]):.3e}')

        return mch
    except Exception as e:
        print(f'ERROR:\n{e}\n')
        return None


##### RUN SCRIPT #####
csv = input("Path to .csv with headers ['path', 'mch_pg'] or click Enter to manually run single dataset:\n")

if not csv == '':
    df = pd.read_csv(csv)
    if not ('path' in df.columns and 'mch_pg' in df.columns):
        raise ValueError(".csv is missing 'path' and/or 'mch_pg' header(s)")

    try:
        savedir = Path(Path(csv).stem)
        os.mkdir("outputs" / savedir)
        # print(f'\nDirectory {savedir} created successfully')
    except FileExistsError:
        # print(f'\nDirectory {savedir} already exists')
        pass

    model = init_model()

    print(f'\n***** Processing batch from: {csv} *****\n')
    batch_mch = [get_dataset_hb(Path(dataset), savedir=savedir) for dataset in tqdm(df['path'].tolist(), desc='Dataset')]
    df['mch_estimate'] = batch_mch

    batch_csv = savedir / f'{savedir}_processed.csv'
    df.to_csv(batch_csv)
    print(f'\nSaved batch mch to {batch_csv}')

else:
    expt = input("Enter single dataset path as per .../LFM_scope/<dataset>/sub_sample_imgs/:\n")
    dataset = Path(f'/hpc/projects/group.bioengineering/LFM_scope/{expt}')

    model = init_model()

    get_dataset_hb(dataset)

