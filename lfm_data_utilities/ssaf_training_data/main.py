import os
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from pathlib import Path
from functools import partial

from lfm_data_utilities.ssaf_training_data import utils

os.environ["MPLBACKEND"] = "Agg"


def process_folder(folder_path: Path, save_loc: Path, focus_graph_loc: Path) -> None:
    """Run the analysis + sorting on a given folder

    Parameters
    ----------
    folder_path: Path
    save_loc: Path
        Where the training data subfolders (i.e displacement folders, [..., +3, +2, ..., -2, -3, ...]) will be saved
    focus_graph_loc: Path
        Where the focus graphs will be saved (optional)
    """

    print(folder_path.stem)

    img_paths = utils.get_list_of_img_paths_in_folder(folder_path)
    motor_positions = utils.get_motor_positions_from_img_paths(img_paths)

    print("Loading images...")
    imgs = utils.load_imgs(img_paths)

    print("Calculating focus metrics...")
    focus_metrics = utils.multiprocess_focus_metric(
        imgs, utils.log_power_spectrum_radial_average_sum
    )

    local_vicinity: int = 10
    max_motor_pos: int = 900
    grouped_focus_metrics = utils.group_by_motor_positions(
        focus_metrics, motor_positions
    )

    # Get mean and normalize focus metric
    metrics_averaged = np.asarray([np.mean(fms) for fms in grouped_focus_metrics])
    metrics_normed = metrics_averaged / np.max(metrics_averaged)

    # Get metrics and motor positions for the values in the local vicinity of the peak focus
    motor_pos_nodup = np.unique(motor_positions)
    peak_focus_pos = np.argmax(metrics_normed).item()

    start = max(peak_focus_pos - local_vicinity, 0)
    end = min(peak_focus_pos + local_vicinity, max_motor_pos)

    motor_pos_local_vicinity = motor_pos_nodup[start:end]
    metrics_local_vicinity = metrics_normed[start:end]

    # Quadratic fit
    qf = np.polynomial.polynomial.Polynomial.fit(
        motor_pos_local_vicinity, metrics_local_vicinity, 2
    )
    curve = qf(motor_pos_local_vicinity)
    peak_focus_motor_position = motor_pos_local_vicinity[np.argmax(curve)]
    peak_motor_pos = peak_focus_motor_position
    grouped_images = utils.group_by_motor_positions(imgs, motor_positions)
    utils.group_by_motor_positions(img_paths, motor_positions)

    n_rows = 4
    n_cols = 3

    predicted_peak = np.argmax(curve) + start
    print(f"predicted peak {predicted_peak=}")

    while True:
        fig = plt.figure(figsize=(4 * 6, 4 * 5))

        gs = fig.add_gridspec(n_rows, n_cols)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            motor_pos_nodup, metrics_normed, label="Focus metric vs. motor position"
        )
        ax0.plot(motor_pos_local_vicinity, curve, label="Curve fit")
        ax0.axvline(peak_motor_pos, color="k", linestyle="--", label="Peak position")
        ax0.plot(
            peak_motor_pos,
            np.max(curve),
            "*",
            label=f"Max@{peak_motor_pos}",
        )

        ax0.set_title(f"{folder_path.stem}")
        ax0.set_xlabel("Motor position (steps)")
        ax0.set_ylabel("Focus metric (dimensionless)")
        ax0.legend()

        ax1 = fig.add_subplot(fig.add_subplot(gs[1, 0]))
        ax1.imshow(grouped_images[predicted_peak][0], cmap="gray")
        ax1.set_title("peak img")
        ax1.axis("off")

        ax2 = fig.add_subplot(fig.add_subplot(gs[1, 1]))
        ax2.imshow(grouped_images[predicted_peak - 1][0], cmap="gray")
        ax2.set_title("peak - 1")
        ax2.axis("off")

        ax3 = fig.add_subplot(fig.add_subplot(gs[1, 2]))
        ax3.imshow(grouped_images[predicted_peak + 1][0], cmap="gray")
        ax3.set_title("peak + 1")
        ax3.axis("off")

        ax4 = fig.add_subplot(fig.add_subplot(gs[2, 0]))
        ax4.imshow(grouped_images[predicted_peak + 2][0], cmap="gray")
        ax4.set_title("peak + 2")
        ax4.axis("off")

        ax5 = fig.add_subplot(fig.add_subplot(gs[2, 1]))
        ax5.imshow(grouped_images[predicted_peak + 3][0], cmap="gray")
        ax5.set_title("peak + 3")
        ax5.axis("off")

        ax5 = fig.add_subplot(fig.add_subplot(gs[2, 2]))
        ax5.imshow(grouped_images[predicted_peak + 4][0], cmap="gray")
        ax5.set_title("peak + 4")
        ax5.axis("off")

        ax6 = fig.add_subplot(fig.add_subplot(gs[3, 0]))
        ax6.imshow(grouped_images[predicted_peak + 5][0], cmap="gray")
        ax6.set_title("peak + 5")
        ax6.axis("off")

        ax7 = fig.add_subplot(fig.add_subplot(gs[3, 1]))
        ax7.imshow(grouped_images[predicted_peak + 6][0], cmap="gray")
        ax7.set_title("peak + 6")
        ax7.axis("off")

        ax8 = fig.add_subplot(fig.add_subplot(gs[3, 2]))
        ax8.imshow(grouped_images[predicted_peak + 7][0], cmap="gray")
        ax8.set_title("peak + 7")
        ax8.axis("off")

        # set tight plot
        fig.tight_layout()

        plt.show()

        input_ = input("Is this peak position correct? (y/n): ")
        if input_ == "y":
            break
        else:
            shift = int(
                input("enter the number of steps to shift the peak position by: ")
            )
            peak_motor_pos += shift

    plt.savefig(f"{focus_graph_loc / folder_path.stem}.png")

    rel_pos = utils.get_relative_to_peak_positions(motor_positions, peak_motor_pos)
    utils.generate_relative_position_folders(save_loc, rel_pos)

    print("Copying images to their relative position folders...")
    utils.move_imgs_to_relative_pos_folders(img_paths, save_loc, rel_pos)


def multiproc_folders(folders: List[Path], save_loc: Path, focus_graph_loc: Path):
    with mp.Pool() as pool:
        tqdm(
            pool.imap(
                partial(
                    process_folder, save_loc=save_loc, focus_graph_loc=focus_graph_loc
                ),
                folders,
            ),
            total=len(folders),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sort zstacks into training data")
    parser.add_argument(
        "unsorted_zstacks_loc",
        type=Path,
        help="Folder path of zstacks to be sorted",
    )
    parser.add_argument(
        "save_loc",
        type=Path,
        help="Folder path where the training data will be saved (can append to folders in an existing training data directory too)",
    )
    parser.add_argument(
        "focus_graph_loc",
        type=Path,
        help="Folder path where the focus graph plots will be saved",
    )
    args = parser.parse_args()

    if not args.save_loc.exists():
        args.save_loc.mkdir(exist_ok=True, parents=True)

    if not args.focus_graph_loc.exists():
        args.focus_graph_loc.mkdir(exist_ok=True, parents=True)

    folders = utils.get_list_of_zstack_folders(args.unsorted_zstacks_loc)

    for folder in folders:
        try:
            process_folder(folder, args.save_loc, args.focus_graph_loc)
        except Exception:
            import traceback

            traceback.print_exc()
