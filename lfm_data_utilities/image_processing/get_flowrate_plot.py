import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import zarr


def downsample_and_slice(
    img: np.ndarray, ds_factor: int, n_slices: int, threshold: bool, thresh_val: int
):
    """Downsample image and return n vertical slices

    Optionally threshold image after downsampling.
    """
    h, w = img.shape
    img_ds = cv2.resize(img, (w // ds_factor, h // ds_factor))

    if threshold and thresh_val is not None:
        img_ds = cv2.threshold(img_ds, thresh_val, 255, cv2.THRESH_BINARY)[1]

    return np.hsplit(img_ds, n_slices)


def run_xcorr(
    zarr_path: Path,
    ds_factor: int = 4,
    n_slices: int = 1,
    threshold: bool = None,
):
    zf = zarr.open(zarr_path, "r")
    h, _ = zf[:, :, 0].shape
    template_vert_offset = int((h // ds_factor) * 0.05)
    corr_coeffs = []
    corr_locs = []

    # Get threshold value peak if necessary
    counts, bins = np.histogram(zf[:, :, 1000], bins=255)
    thresh_val = bins[np.argmax(counts)]

    for i in tqdm(range(0, zf.initialized - 1)):
        img = zf[:, :, i]
        img2 = zf[:, :, i + 1]
        slices1 = downsample_and_slice(img, ds_factor, n_slices, threshold, thresh_val)
        slices2 = downsample_and_slice(img2, ds_factor, n_slices, threshold, thresh_val)

        # Get xcorr between each pair of slices
        for s1, s2 in zip(slices1, slices2):
            res = cv2.matchTemplate(
                s2, s1[:template_vert_offset, :], cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            corr_coeffs.append(max_val)
            corr_locs.append(max_loc[1])
    return corr_coeffs, corr_locs


def plot(no_thresh_coeffs, no_thresh_locs, thresh_coeffs, thresh_locs):
    fig, ax = plt.subplots(2, 2, figsize=(12, 5))
    bin_width = 0.01
    bins = np.arange(
        np.min([no_thresh_coeffs, thresh_coeffs]),
        np.max([no_thresh_coeffs, thresh_coeffs]) + bin_width,
        bin_width,
    )

    ax[0, 0].axhline(0)
    ax[0, 0].hist(no_thresh_coeffs, bins=bins, label="Original", edgecolor="k")
    ax[0, 0].hist(
        thresh_coeffs,
        bins=bins,
        weights=-np.ones_like(thresh_coeffs),
        label="Thresholded",
        edgecolor="k",
    )
    ax[0, 0].legend()
    ax[0, 0].set_title("xcorr match value")
    ax[0, 0].yaxis.set_major_formatter(lambda x, pos: f"{abs(x):g}")

    bin_width = 2
    bins = np.arange(
        np.min([no_thresh_locs, thresh_locs]),
        np.max([no_thresh_locs, thresh_locs]) + bin_width,
        bin_width,
    )
    ax[0, 1].axhline(0)
    ax[0, 1].hist(no_thresh_locs, bins=bins, label="Original", edgecolor="k")
    ax[0, 1].hist(
        thresh_locs,
        bins=bins,
        weights=-np.ones_like(thresh_locs),
        label="Thresholded",
        edgecolor="k",
    )
    ax[0, 1].legend()
    ax[0, 1].set_title("xcorr dy vals")
    ax[0, 1].yaxis.set_major_formatter(lambda x, pos: f"{abs(x):g}")

    def moving_average(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    no_thresh_moving_avg = moving_average(no_thresh_locs, 300)
    thresh_moving_avg = moving_average(thresh_locs, 300)

    ymin, ymax = np.min([no_thresh_moving_avg, thresh_moving_avg]), np.max(
        [no_thresh_moving_avg, thresh_moving_avg]
    )

    ax[1, 0].plot(no_thresh_moving_avg, "o", markersize=0.5)
    ax[1, 0].set_title("xcorr dy, moving average")
    ax[1, 0].set_ylim(ymin, ymax)

    ax[1, 1].plot(thresh_moving_avg, "o", color="C1", markersize=0.5)
    ax[1, 1].set_title("xcorr dy, moving average, thresholded images")
    ax[1, 1].set_ylim(ymin, ymax)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FlowratePlots",
        description="Read in a zarr dataset and run cross correlation on all adjacent image pairs.",
    )

    parser.add_argument("-zarr", "--zarr", help="Path to zarr .zip file", type=Path)

    args = parser.parse_args()

    if not (Path(args.zarr).exists()):
        raise ValueError(f"This file does not exist: {args.zarr}")

    print(
        "Running xcorr on full dataset twice - once without threhsolding images, once with."
    )
    no_thresh_coeffs, no_thresh_locs = run_xcorr(args.zarr)
    thresh_coeffs, thresh_locs = run_xcorr(args.zarr, threshold=True)

    plot(no_thresh_coeffs, no_thresh_locs, thresh_coeffs, thresh_locs)
