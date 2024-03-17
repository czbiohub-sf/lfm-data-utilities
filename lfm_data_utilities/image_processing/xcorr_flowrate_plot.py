import argparse
from pathlib import Path
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import zarr


def get_ewma(data, alpha=0.1):
    prev = data[0]
    ewma_vals = [prev]
    for v in data[1:]:
        new_val = prev * (1 - alpha) + v * alpha
        ewma_vals.append(new_val)
        prev = new_val
    return ewma_vals


def downsample_and_slice(img: np.ndarray, ds_factor: int, n_slices: int):
    """Downsample image and return n vertical slices"""

    h, w = img.shape
    img_ds = cv2.resize(img, (w // ds_factor, h // ds_factor))

    return np.hsplit(img_ds, n_slices)


def run_xcorr(
    zarr_path: Path,
    ds_factor: int = 1,
    n_slices: int = 1,
):
    zf = zarr.open(zarr_path, "r")
    h, _ = zf[:, :, 0].shape
    template_vert_offset = int((h // ds_factor) * 0.05)
    corr_coeffs = []
    corr_locs = []

    for i in tqdm(range(0, zf.initialized - 1)):
        img = zf[:, :, i]
        img2 = zf[:, :, i + 1]
        slices1 = downsample_and_slice(img, ds_factor, n_slices)
        slices2 = downsample_and_slice(img2, ds_factor, n_slices)

        # Get xcorr between each pair of slices
        for s1, s2 in zip(slices1, slices2):
            res = cv2.matchTemplate(
                s2, s1[:template_vert_offset, :], cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            corr_coeffs.append(max_val)
            corr_locs.append(max_loc[1])
    return corr_coeffs, corr_locs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple flowrate plot",
        description="Given a zarr file and a downsampling factor, save a plot of the xcorr y diffs",
    )

    parser.add_argument("zarr_path", type=Path, help="Path to the zarr file")
    parser.add_argument(
        "downsample_factor", type=int, help="Downsampling factor", default=1
    )
    parser.add_argument("save_loc", type=Path, help="Path to save the plot")

    args = parser.parse_args()
    ds_factor = args.downsample_factor

    # Get xcorr diffs
    coeffs, locs = run_xcorr(args.zarr_path, ds_factor=ds_factor, n_slices=1)
    locs = np.asarray(locs)
    m, sd = np.mean(locs) * ds_factor, np.std(locs) * ds_factor
    ewma_alpha = 0.05

    # Plot
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"{Path(args.zarr_path).stem}")
    plt.plot(locs * ds_factor, "o", markersize=0.5, alpha=0.5, label="Raw")
    plt.plot(
        get_ewma(locs * ds_factor, ewma_alpha),
        alpha=0.75,
        label=f"EWMA, alpha={ewma_alpha}",
    )
    plt.title(
        f"Downsampled {ds_factor}x xCorr y-diff vs. frame\nm={m:.2f}, sd={sd:.2f}"
    )
    plt.xlabel("Frame idx")
    plt.ylabel("Displacement (pixels)")
    plt.ylim(0, 772)
    plt.legend()

    plt.savefig(f"{args.save_loc}/{Path(args.zarr_path).stem}_xcorr_ds{ds_factor}.png")
