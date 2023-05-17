import os
import argparse
import multiprocessing as mp

from tqdm import tqdm
from typing import List
from pathlib import Path
from functools import partial

from lfm_data_utilities.ssaf_training_data import utils

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


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

    peak_motor_pos = utils.find_peak_position(
        focus_metrics,
        motor_positions,
        save_loc=focus_graph_loc,
        folder_name=folder_path.stem,
    )

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
