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


def get_diffs_from_matches(matches, kp, t_kp):
    x_diffs = []
    y_diffs = []
    for match in matches:
        p1 = kp[match.queryIdx].pt
        p2 = t_kp[match.trainIdx].pt
        x_diffs.append(p2[0] - p1[0])
        y_diffs.append(p2[1] - p1[1])
    return np.asarray(x_diffs), np.asarray(y_diffs)


def get_orb_xy_diffs(zf_path: Path, scale_factor: int = 1):
    num_features = 500
    zf = zarr.open(zf_path, "r")
    orb = cv2.ORB_create(num_features)

    h, w = zf[:, :, 0].shape

    matcher = cv2.BFMatcher()

    x_diff_pointer = 0
    y_diff_pointer = 0
    all_x_diffs = np.zeros(num_features * zf.initialized)
    all_y_diffs = np.zeros(num_features * zf.initialized)

    pos_y_diff_means = np.zeros(zf.initialized)

    for i in tqdm(range(1, zf.initialized)):
        i1 = cv2.resize(zf[:, :, i - 1], (w // scale_factor, h // scale_factor))
        i2 = cv2.resize(zf[:, :, i], (w // scale_factor, h // scale_factor))

        kp, des = orb.detectAndCompute(i1, None)
        t_kp, t_des = orb.detectAndCompute(i2, None)

        matches = matcher.match(des, t_des)

        x_diffs, y_diffs = np.asarray(get_diffs_from_matches(matches, kp, t_kp))

        if len(y_diffs[y_diffs > 0]) > 0:
            pos_mean = np.mean(y_diffs[y_diffs > 0])
        else:
            pos_mean = 0

        all_x_diffs[x_diff_pointer : x_diff_pointer + len(x_diffs)] = x_diffs
        all_y_diffs[y_diff_pointer : y_diff_pointer + len(y_diffs)] = y_diffs
        x_diff_pointer += len(x_diffs)
        y_diff_pointer += len(y_diffs)

        pos_y_diff_means[i] = pos_mean

    return all_x_diffs, all_y_diffs, pos_y_diff_means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple ORB plot",
        description="Given a zarr file and a downsampling factor, save a plot of the ORB y diffs",
    )

    parser.add_argument("zarr_path", type=Path, help="Path to the zarr file")
    parser.add_argument(
        "downsample_factor", type=int, help="Downsampling factor", default=1
    )
    parser.add_argument("save_loc", type=Path, help="Path to save the plot")

    args = parser.parse_args()
    ds_factor = args.downsample_factor

    # Get orb diffs
    all_x_diffs, all_y_diffs, pos_y_diff_means = get_orb_xy_diffs(
        args.zarr_path, scale_factor=ds_factor
    )
    m, sd = np.mean(pos_y_diff_means) * ds_factor, np.std(pos_y_diff_means) * ds_factor

    ewma_alpha = 0.05

    # Plot
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"{Path(args.zarr_path).stem}")
    plt.plot(pos_y_diff_means * ds_factor, "o", markersize=0.5, alpha=0.5, label="Raw")
    plt.plot(
        get_ewma(pos_y_diff_means * ds_factor, ewma_alpha),
        alpha=0.75,
        label=f"EWMA, alpha={ewma_alpha}",
    )
    plt.title(
        f"Downsampled {ds_factor}x ORB positive y feature diffs vs. frame\nm={m:.2f}, sd={sd:.2f}"
    )
    plt.xlabel("Frame idx")
    plt.ylabel("Displacement (pixels)")
    plt.ylim(0, 772)
    plt.legend()

    plt.savefig(f"{args.save_loc}/{Path(args.zarr_path).stem}_orb_ds{ds_factor}.png")
