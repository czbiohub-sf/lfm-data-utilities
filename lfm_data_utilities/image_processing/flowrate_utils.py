from typing import Tuple, List
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from lfm_data_utilities.utils import get_all_dataset_paths, Dataset


def downSampleImage(img: np.ndarray, scale_factor: int) -> np.ndarray:
    """Downsamples an image by `scale_factor`"""
    h, w = img.shape
    return cv2.resize(img, (w // scale_factor, h // scale_factor))


def getTemplateRegion(
    img: np.ndarray,
    x1_perc: float = 0.05,
    y1_perc: float = 0.05,
    x2_perc: float = 0.45,
    y2_perc: float = 0.85,
):
    """Returns a subregion of the image provided.
    The start and end positions are to be given as percentages of the image's shape.

    Parameters
    ----------
        img : np.ndarray
            An image (i.e a numpy array)
        x1_perc : float
            What percentage of the image the start of the subregion should begin at.
            For example, x1_perc=0.05 of an input image with shape (600, 800) would mean the
            subregion would begin at 0.05*600=30.
        y1_perc: float
            See x1_perc
        x2_perc: float
            See x1_perc
        y2_perc: float
            See x1_perc

    Returns
    -------
        np.ndarray:
            subregion of the input array
        int:
            x_offset (i.e x start of the subregion)
        int:
            y_offset (i.e y start of the subregion)
    """

    h, w = img.shape
    xs, xf = int(x1_perc * w), int(x2_perc * w)
    ys, yf = int(y1_perc * h), int(y2_perc * h)

    return img[ys:yf, xs:xf], (xs, xf), (ys, yf)


def get_flowrate_with_cross_correlation(
    prev_img: np.ndarray,
    next_img: np.ndarray,
    scale_factor: int = 10,
    temp_x1_perc: float = 0.05,
    temp_y1_perc: float = 0.05,
    temp_x2_perc: float = 0.45,
    temp_y2_perc: float = 0.85,
    debug: bool = False,
) -> Tuple[float, float, float]:
    """Find the displacement of a subregion of an image with another, temporally adjacent, image.

    Parameters
    ----------
        prev_img : np.ndarray
            First image
        next_img: np.ndarray
            Subsequent imag
        scale_factor : int
            Factor to use for downsampling the images
        temp_x1_perc : float
            What percentage of the image the start of the subregion should begin at.
            For example, x1_perc=0.05 of an input image with shape (600, 800) would mean the
            subregion would begin at 0.05*600=30.
        temp_y1_perc: float
            See x1_perc
        temp_x2_perc: float
            See x1_perc
        temp_y2_perc: float
            See x1_perc

    Returns
    -------
        int:
            dx: displacement in x
        int:
            dy: displacement in y
    """
    im1_ds, im2_ds = downSampleImage(prev_img, scale_factor), downSampleImage(
        next_img, scale_factor
    )

    # Select the subregion within the first image by defining which quantiles to use
    im1_ds_subregion, x_offset, y_offset = getTemplateRegion(
        im1_ds, 0.05, 0.05, 0.85, 0.45
    )

    # Run a normalized cross correlation between the image to search and subregion
    template_result = cv2.matchTemplate(im2_ds, im1_ds_subregion, cv2.TM_CCOEFF_NORMED)

    # Find the point with the maximum value (i.e highest match) and caclulate the displacement
    _, max_val, _, max_loc = cv2.minMaxLoc(template_result)
    dx, dy = max_loc[0] - x_offset[0], max_loc[1] - y_offset[0]

    # If debug mode is on, run `plot_cc` which saves images of the cross-correlation calculation.
    if debug:
        plot_cc(
            im1_ds,
            im2_ds,
            im1_ds_subregion,
            template_result,
            (x_offset[0], y_offset[0]),
            (x_offset[1], y_offset[1]),
            max_loc[0],
            max_loc[1],
            dx,
            dy,
        )

    return float(dx), float(dy), float(max_val)


def plot_cc(im1, im2, im1_subregion, template_result, xy1, xy2, max_x, max_y, dx, dy):
    """A function for debugging and visualizing the cross-correlation
    displacement calculation.

    This function produces a 2x2 plot of:
        Top-left: The first image
        Bottom-left: The second image
        Top-right: A close-up of the subregion of interest in the first image
        Bottom-right: A close-up of the subregion with the closest match in the second image
    """

    h, w = im1_subregion.shape
    im1_subregion = im1_subregion.copy()
    im2_subregion = im2[max_y : max_y + h, max_x : max_x + w].copy()
    im1 = cv2.rectangle(im1, xy1, xy2, 255, 1)
    im2 = cv2.rectangle(im2, (max_x, max_y), (max_x + w, max_y + h), 255, 1)

    fig, ax = plt.subplots(3, 2, figsize=(10, 7))
    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, :])

    ax1.imshow(im1, cmap="gray")
    ax2.imshow(im1_subregion, cmap="gray")
    ax3.imshow(im2, cmap="gray")
    ax4.imshow(im2_subregion, cmap="gray")
    ax4.text(0, 0, f"{dx, dy}", bbox={"facecolor": "white", "pad": 2})
    ax5.imshow(template_result)
    plt.show(block=False)
    plt.pause(0.01)
    plt.tight_layout()
    c = input("Press enter to advance: ")
    if c == "c":
        exit()
    plt.close()


def get_all_flowrates_from_experiment(
    top_level_dir: Path, verbose: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """Return the cross correlation values between all pairs of adjacent images in the given zarr file.

    Parameters
    ----------
    top_level_dir: Path
        Experiment folder (i.e the folder containing the zarr file and metadata files)

    Returns
    ------
    Tuple [ List[float], List[float], List[float] ]
        dx, dy, confidences
    """

    dataset = Dataset(get_all_dataset_paths(top_level_dir)[0])
    zf = dataset.zarr_file
    per_img_csv = dataset.per_img_metadata

    h, w, _ = zf.shape
    num_images: int = zf.initialized
    time_diffs = np.diff([float(x) for x in per_img_csv["vals"]["timestamp"]])
    scale_factor = 10

    dx_vals = []
    dy_vals = []
    conf_vals = []

    for i in tqdm(range(1, num_images), disable=not verbose):
        prev_img = zf[:, :, i - 1]
        img = zf[:, :, i]
        dx, dy, max_conf = get_flowrate_with_cross_correlation(
            prev_img=prev_img, next_img=img, scale_factor=scale_factor
        )

        dt = time_diffs[i - 1]
        dx = (dx / dt) / (w / scale_factor)
        dy = (dy / dt) / (h / scale_factor)

        dx_vals.append(dx)
        dy_vals.append(dy)
        conf_vals.append(max_conf)

    return (dx_vals, dy_vals, conf_vals)
