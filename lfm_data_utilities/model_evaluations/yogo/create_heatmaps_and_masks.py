#! /usr/bin/env python3

"""
Run YOGO on the given dataset(s) and for each, return a heatmap of
where each class is detected (spatially).

The heatmap is saved as a `.npy` file, array of shape (Sx * Sy * NUM_CLASSES),
where Sx and Sy are the shape of YOGO's output grid size. At the time of writing, Sx = 129, Sy = 97.
"""

from pathlib import Path
from typing import Union
import math

import cv2
import zarr
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from scipy import ndimage
from yogo.model import YOGO


def load_model(path_to_pth_file: str, device: Union[str, torch.device] = "cpu") -> YOGO:
    """Convenience function to load model.

    Parameters
    ----------
    path_to_pth_file: str
        Model's .pth filepath
    device: str / torch.device
        Either cpu / gpu

    Returns
    -------
    YOGO
    """
    model, _ = YOGO.from_pth(Path(path_to_pth_file), inference=True)
    model.to(device)
    model.eval()

    return model


def get_img_from_zarr_in_torch_format(zf: zarr.core.Array, id: int) -> torch.Tensor:
    """Convenience function for loading an image from zarr in a format
    ready for GPU-inference.

    Parameters
    ----------
    zf: zarr.core.Array (i.e the zarr file, zarr.open("filename", "r"))
    id: int
        Which image to pull from the zarr (zf[:, :, id])

    Returns
    -------
    torch.Tensor
    """

    assert (
        torch.cuda.is_available()
    ), "GPU unavailable, can't convert image to GPU ready format"
    return torch.unsqueeze(torch.tensor(zf[:, :, id]).to(device="cuda"), 0)


def circ_kernel(radius):
    """Create a circular mask with the given radius.

    Parameters
    ----------
    radius: int

    Returns
    -------
    np.ndarray
    """

    xx, yy = np.mgrid[:radius, :radius]
    circ = (xx - radius // 2) ** 2 + (yy - radius // 2) ** 2
    return circ <= radius


def generate_heatmap(
    zf: zarr.core.Array,
    model: YOGO,
    sx: int = 129,
    sy: int = 97,
    num_classes: int = 7,
    thresh: float = 0.9,
) -> np.ndarray:
    """
    Create a heatmap for each class for the given dataset.

    Parameters
    ----------
    zf: zarr.core.Array
        Zarr dataset (i.e the result of zarr.open(filename, "r"))
    model: YOGO
    sx: int = 129
    sy: int = 97
    thresh: float=0.9

    Returns
    -------
    maps: np.ndarray
    """

    maps = np.zeros((sy, sx, num_classes))
    for i in tqdm(range(zf.initialized)):
        img = get_img_from_zarr_in_torch_format(zf, i)
        pred = model(img).squeeze()
        for i in range(num_classes):
            thresh_mask = pred[5 + i, :, :] >= thresh
            pred[5 + i, ~thresh_mask] = 0
            maps[:, :, i] += pred[5 + i, :, :].detach().cpu().numpy()

    return maps.astype(np.uint16)


def generate_masks(
    heatmaps: np.ndarray,
    sx: int = 129,
    sy: int = 97,
    num_classes: int = 7,
) -> np.ndarray:
    """
    Heatmaps should be a numpy array of shape (sx * sx * NUM_CLASSES). The By default, sy and sx are 97, 129

    This function applies Otsu or standard-deviation thresholding and returns a map for each class, as a single array.
    Only parasite classes will be masked - the masks for healthy/wbc/misc will just be a mask of ones and can be ignored.

    Parameters
    ----------
    heatmaps: np.ndarray
        Array of shape sx, sy, num_classes
    sx: int = 129
        YOGO grid size, x
    sy: int = 97
        YOGO grid size, y
    num_classes: int = 7

    Returns
    -------
    np.ndarray, dtype: bool
        sx, sy, num_classes - binary values (1/0).
        If 1, mask OUT that grid spot (i.e don't keep it). If 0, keep that grid spot.
    """

    masks = np.zeros((sy, sx, num_classes))
    inset_offset = 5  # Crop out the edges of the heatmap before finding threshold value

    parasite_classes = [1, 2, 3, 4]  # Ring, troph, schizont, gametocyte

    dilation_kernel = circ_kernel(3)

    for idx in parasite_classes:
        heatmap = heatmaps[:, :, idx]
        mean, sd = np.mean(heatmap), np.std(heatmap)
        thresh, _ = cv2.threshold(
            heatmap[inset_offset:-inset_offset, inset_offset:-inset_offset],
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        thresh = max(thresh, mean + 3 * sd, 2)
        mask = heatmap > thresh
        dilated = ndimage.binary_dilation(mask.astype(bool), dilation_kernel)
        masks[:, :, idx] = dilated
    return masks.astype(bool)


def create_and_save_heatmap_and_mask_plot(
    heatmap: np.ndarray,
    mask: np.ndarray,
    fn: Path,
) -> None:
    fig = plt.figure(figsize=(6, 12))
    gs = gridspec.GridSpec(4, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])

    ax1.set_title("Ring heatmap")
    ax1.imshow(heatmap[:, :, 1])
    ax2.set_title("Ring hotspot mask")
    ax2.imshow(mask[:, :, 1])

    ax3.set_title("Troph heatmap")
    ax3.imshow(heatmap[:, :, 2])
    ax4.set_title("Troph hotspot mask")
    ax4.imshow(mask[:, :, 2])

    ax5.set_title("Schizont heatmap")
    ax5.imshow(heatmap[:, :, 3])
    ax6.set_title("Schizont hotspot mask")
    ax6.imshow(mask[:, :, 3])

    ax7.set_title("Gametocyte heatmap")
    ax7.imshow(heatmap[:, :, 4])
    ax8.set_title("Gametocyte hotspot mask")
    ax8.imshow(mask[:, :, 4])

    plt.tight_layout()
    _ = plt.suptitle(f"{fn.stem} - heatmaps and masks")
    fig.subplots_adjust(top=0.95)
    plt.savefig(f"{fn}")
    plt.close()


def vertical_crop(masks, crop_perc: float = 1, h: int = 97):
    if crop_perc == 1:
        return masks

    center_y = h // 2
    lower_lim = int(center_y - (crop_perc / 2 * h))
    upper_lim = int(center_y + (crop_perc / 2 * h))

    return masks[lower_lim:upper_lim, :, :]


def create_heatmaps_and_masks(
    path_to_pth: Path,
    target_dataset: Path,
    heatmaps_dir: Path,
    masks_dir: Path,
    plots_dir: Path,
    overwrite_existing: bool = False,
    thresh: float = 0.9,
    crop_height: float = 1,
) -> None:
    print(f"Working on {target_dataset}")
    filename = target_dataset.stem + ".npy"
    plot_filename = plots_dir / (target_dataset.stem + ".jpg")
    if plot_filename.exists() and not overwrite_existing:
        print(f"{plot_filename} already created, skipping!")
        return

    print(f"Loading model: {path_to_pth}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(path_to_pth), device)
    try:
        zf = zarr.open(target_dataset, "r")
    except Exception as e:
        print(
            f"Failed to open target dataset - most likely corrupt; {target_dataset}: {e}"
        )
        return

    if not zf.initialized > 0:
        print(f"Empty dataset - {target_dataset}")
        return

    heatmap = generate_heatmap(zf, model, thresh=thresh)
    mask = generate_masks(heatmap)
    cropped_mask = vertical_crop(mask, crop_height)

    np.save(heatmaps_dir / filename, heatmap)
    np.save(masks_dir / filename, cropped_mask)
    create_and_save_heatmap_and_mask_plot(heatmap, mask, plot_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate heatmap masks")
    parser.add_argument("path_to_pth", type=Path, help="Path to YOGO .pth file")
    parser.add_argument(
        "save_dir", type=Path, help="Where to save the plots and .npy files"
    )
    parser.add_argument(
        "--overwrite-existing",
        default=False,
        help="Overwrite existing npy files (defaults to False)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.9,
        help="Confidence threshold for heatmaps",
    )
    parser.add_argument(
        "--vertical-crop-height", type=float, default=1, help="Vertical crop height"
    )

    meg = parser.add_mutually_exclusive_group(required=True)
    meg.add_argument("--target-zip", type=Path, help="Path to zarr (.zip) file")
    meg.add_argument(
        "--list-of-target-zip",
        type=Path,
        help="Path to file with one zip file on each line",
    )

    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True, parents=True)

    heatmaps_dir = args.save_dir / "heatmaps_npy"
    masks_dir = args.save_dir / "masks_npy"
    plots_dir = args.save_dir / "plots"

    for x in (heatmaps_dir, masks_dir, plots_dir):
        x.mkdir(exist_ok=True, parents=True)

    if args.target_zip:
        create_heatmaps_and_masks(
            args.path_to_pth,
            args.target_zip,
            heatmaps_dir,
            masks_dir,
            plots_dir,
            thresh=args.conf_thresh,
            overwrite_existing=args.overwrite_existing,
            crop_height=args.vertical_crop_height,
        )
    elif args.list_of_target_zip:
        with open(args.list_of_target_zip, "r") as f:
            zip_paths = [Path(x.strip()) for x in f.readlines()]

        for zip_path in zip_paths:
            create_heatmaps_and_masks(
                args.path_to_pth,
                zip_path,
                heatmaps_dir,
                masks_dir,
                plots_dir,
                thresh=args.conf_thresh,
                overwrite_existing=args.overwrite_existing,
                crop_height=args.vertical_crop_height,
            )
