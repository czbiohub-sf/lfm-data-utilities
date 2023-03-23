from functools import partial

from tqdm import tqdm

from lfm_data_utilities.ssaf_training_data.utils import *


def process_folder(
    folder_path: Path, save_loc: Path, focus_graph_loc: Path = None
) -> None:
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

    img_paths = get_list_of_img_paths_in_folder(folder_path)
    motor_positions = get_motor_positions_from_img_paths(img_paths)

    print("Loading images...")
    imgs = load_imgs(img_paths)

    print("Calculating focus metrics...")
    focus_metrics = multiprocess_focus_metric(
        imgs, log_power_spectrum_radial_average_sum
    )

    peak_motor_pos = find_peak_position(
        focus_metrics,
        motor_positions,
        save_loc=focus_graph_loc,
        folder_name=folder_path.stem,
    )

    rel_pos = get_relative_to_peak_positions(motor_positions, peak_motor_pos)
    generate_relative_position_folders(save_loc, rel_pos)

    print("Copying images to their relative position folders...")
    move_imgs_to_relative_pos_folders(img_paths, save_loc, rel_pos)


def multiproc_folders(folders: List[Path], save_loc: Path, focus_graph_loc: Path):
    with Pool() as pool:
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

    tld = Path(input("Folder path of zstacks to be sorted: "))
    save_loc = Path(
        input(
            "Folder path where the training data will be saved (can append to folders in an existing training data directory too): "
        )
    )
    focus_graph_loc = Path(
        input("Folder path where the focus graph plots will be saved: ")
    )

    if not save_loc.exists():
        Path.mkdir(save_loc)

    if not focus_graph_loc.exists():
        Path.mkdir(focus_graph_loc)

    folders = get_list_of_zstack_folders(tld)
    valid_folders = get_valid_folders(folders)

    # multiproc_folders(valid_folders, save_loc, focus_graph_loc)
    for folder in valid_folders:
        process_folder(folder, save_loc, focus_graph_loc)
