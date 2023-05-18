import sys
import random
import termios
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path

from lfm_data_utilities.ssaf_training_data import utils


# (Axel) I am not proud of this function - it's messy and hacky and fragile,
# but i don't want to spend a lot of time working on this, so hacky and fast
# is the way to go
def process_folder(folder_path: Path, save_loc: Path, focus_graph_loc: Path):
    """Run the analysis + sorting on a given folder

    Parameters
    ----------
    folder_path: Path
    save_loc: Path
        Where the training data subfolders (i.e displacement folders, [..., +3, +2, ..., -2, -3, ...]) will be saved
    focus_graph_loc: Path
        Where the focus graphs will be saved (optional)
    """
    local_vicinity: int = 10
    max_motor_pos: int = 900

    img_paths = utils.get_list_of_img_paths_in_folder(folder_path)
    motor_positions = utils.get_motor_positions_from_img_paths(img_paths)
    motor_pos_nodup = np.unique(motor_positions)

    print("Loading images...")
    imgs = utils.load_imgs(img_paths)
    grouped_images = utils.group_by_motor_positions(imgs, motor_positions)

    print("Calculating focus metrics...")
    focus_metrics = utils.multiprocess_focus_metric(
        imgs, utils.log_power_spectrum_radial_average_sum
    )

    grouped_focus_metrics = utils.group_by_motor_positions(
        focus_metrics, motor_positions
    )

    # Get mean and normalize focus metric
    metrics_averaged = np.asarray([np.mean(fms) for fms in grouped_focus_metrics])
    metrics_normed = metrics_averaged / np.max(metrics_averaged)

    # Get metrics and motor positions for the values in the local vicinity of the peak focus
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

    n_rows, n_cols = 3, 3

    predicted_peak = np.argmax(curve).item() + start

    while True:
        fig = plt.figure(figsize=(6 * 6, 6 * 5), layout="constrained")
        fig.set_facecolor("gray")

        gs = fig.add_gridspec(n_rows, n_cols)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            motor_pos_nodup, metrics_normed, label="Focus metric vs. motor position"
        )
        ax0.plot(motor_pos_local_vicinity, curve, label="Curve fit")
        ax0.plot(
            peak_focus_motor_position,
            np.max(curve),
            "*",
            label=f"Max@{peak_focus_motor_position}",
        )
        ax0.axvline(
            motor_pos_nodup[predicted_peak],
            color="k",
            linestyle="--",
            label="set zero for zstack",
        )
        ax0.tick_params(axis="y", direction="in", pad=-22)
        ax0.tick_params(axis="x", direction="in", pad=-40)

        # ax0.xaxis.set_label_position('top')
        # ax0.set_xlabel("Motor position (steps)")
        ax0.xaxis.tick_top()
        ax0.set_title(f"{folder_path.stem}")
        ax0.legend()

        ax1 = fig.add_subplot(fig.add_subplot(gs[1, 0]))
        ax1.imshow(random.choice(grouped_images[predicted_peak - 1]), cmap="gray", vmin=0, vmax=255, interpolation=None)
        ax1.set_title("focus - 1 step")
        ax1.axis("off")

        ax2 = fig.add_subplot(fig.add_subplot(gs[2, 0]))
        ax2.imshow(random.choice(grouped_images[predicted_peak - 2]), cmap="gray", vmin=0, vmax=255, interpolation=None)
        ax2.set_title("focus - 2 steps")
        ax2.axis("off")

        ax3 = fig.add_subplot(fig.add_subplot(gs[1, 1]))
        ax3.imshow(random.choice(grouped_images[predicted_peak]), cmap="gray", vmin=0, vmax=255, interpolation=None)
        ax3.set_title("focus")
        ax3.axis("off")

        ax4 = fig.add_subplot(fig.add_subplot(gs[2, 1]))
        ax4.imshow(random.choice(grouped_images[predicted_peak]), cmap="gray", vmin=0, vmax=255, interpolation=None)
        ax4.set_title("focus (another one)")
        ax4.axis("off")

        ax5 = fig.add_subplot(fig.add_subplot(gs[1, 2]))
        ax5.imshow(random.choice(grouped_images[predicted_peak + 1]), cmap="gray", vmin=0, vmax=255, interpolation=None)
        ax5.set_title("focus + 1 step")
        ax5.axis("off")

        ax6 = fig.add_subplot(fig.add_subplot(gs[2, 2]))
        ax6.imshow(random.choice(grouped_images[predicted_peak + 2]), cmap="gray", vmin=0, vmax=255, interpolation=None)
        ax6.set_title("focus + 2 steps")
        ax6.axis("off")

        gs.tight_layout(fig)

        plt.show()

        try:
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        except Exception as e:
            raise NotImplementedError(
                "most likely, you are running Windows, which we do not support "
                "(and also our software isn't compatible with Windows)"
            ) from e

        input_ = input(
            "Is this peak position correct? ('y' for yes, 'n' for no, 's' to skip folder, and 'enter' to see different images): "
        )
        if input_ == "y":
            break
        elif input_ == "n":
            shift = 0
            while True:
                try:
                    usr_input = input(
                        "enter the number of steps to " "shift the peak position by: "
                    )
                    shift = int(usr_input)
                    break
                except (TypeError, ValueError):
                    print(f"{usr_input} doesn't seem to be an integer; try again")

            predicted_peak += shift
        elif input_ == "s":
            return None
        else:
            continue

    peak_focus_motor_position = motor_pos_nodup[predicted_peak]

    plt.savefig(f"{focus_graph_loc / folder_path.stem}.png")

    rel_pos = utils.get_relative_to_peak_positions(
        motor_positions, peak_focus_motor_position
    )
    utils.generate_relative_position_folders(save_loc, rel_pos)

    return img_paths, save_loc, rel_pos


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

    input(
        "\n\nThis will take you through z-stacks and will prompt you for input.\n"
        "We will load a set of images and show you a focus plot, along with a set of images.\n"
        "Once you close the plot, you will be promped on whether the focus plot was correct.\n"
        "If it was correct, this will start sorting the images in the background and go to the next z-stack.\n"
        "If it was not correct, this will prompt you for the correction in steps (eg '1', '-1', '2', etc)\n"
        "If you can't tell, when promped, just press enter to see more images.\n"
        "You can also skip a folder to not sort it at all, e.g. if you've already sorted it or if it's just too bad to sort.\n"
        "Make sure that the images marked as 'focus' are actually in focus.\n"
        "At the end, there will be a wait to let the program finish sorting images. Go get a coffee, and revel in the glory of a human-machine partnership!\n\n"
        "press enter to continue...\n"
    )

    procs = []
    for i, folder in enumerate(folders, start=1):
        try:
            print("-----------------------------")
            print(f"{folder.name}")
            print(f"folder {i} / {len(folders)}")
            ret_val = process_folder(
                folder, args.save_loc, args.focus_graph_loc
            )
            if ret_val is None:
                continue

            img_paths, save_loc, rel_pos = ret_val

            print(
                "Copying images to their relative position folders in child process..."
            )

            proc = mp.Process(
                target=utils.move_imgs_to_relative_pos_folders,
                args=(img_paths, save_loc, rel_pos),
            )
            proc.start()
            procs.append(proc)
        except Exception:
            import traceback

            traceback.print_exc()

    print(
        "waiting for images to finish copying to their folders - this can take a while..."
    )
    for p in procs:
        p.join()
