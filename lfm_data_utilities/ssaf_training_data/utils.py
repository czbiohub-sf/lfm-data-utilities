import cv2
import numpy as np


from tqdm import tqdm
from os import listdir
from collections import defaultdict
from typing import List, Callable, Any
from shutil import copy
from pathlib import Path
from multiprocessing import Pool
from lfm_data_utilities.utils import PathLike


def get_list_of_zstack_folders(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all "local_zstack" subfolders

    Parameters
    ----------
    top_level_dir: str
        Top level directory path

    Returns
    -------
    List[Path]
        All subfolders within the given top level directory which contain ZStacks
        A folder is considered to be a zstack folder if it contains "local_zstack" in its name.
    """

    return sorted(
        [f for f in Path(top_level_dir).glob("**/*local_zstack*/") if f.name[0] != "."]
    )


def get_valid_folders(
    folder_list: List[Path], imgs_per_step: int = 30, num_motor_steps: int = 60
) -> List[Path]:
    """Return valid folders from a given list of folder paths.

    Valid folders are one which contain the expected number of images given the imgs taken per step
    and the total number of motor steps taken.

    This function additionally prints out the name of the folders which did not have the expected number of images.

    Parameters
    ----------
    folder_list: List[Path]
        List of folder paths
    imgs_per_step: int=30
        Number of images taken at each motor step
    num_motor_steps: int=30
        Sweep range of the motor

    Returns
    -------
    List[Path]
        The set of folders from the original list which have the expected number of images
    """

    expected_num_imgs = imgs_per_step * num_motor_steps
    valid_folders = [f for f in folder_list if len(listdir(f)) == expected_num_imgs]

    for f in set(folder_list) - set(valid_folders):
        print(
            f"Folder {f.name} did not contain the expected number of images. Expected: {expected_num_imgs}, actual: {len(listdir(f))}"
        )

    return valid_folders


def get_motor_position_from_path(path: Path) -> int:
    """Parse and return the motor position from the filepath.

    Parameters
    ----------
    path: str
        Image filepath
    Returns
    -------
    int
        Motor position where the image was taken
    """

    return int(path.stem.split("_")[0])


def get_image_step_idx_from_path(path: Path) -> int:
    """Parse and return the image step index from the filepath.

    Parameters
    ----------
    path: str
        Image filepath
    Returns
    -------
    int
        Motor position where the image was taken
    """

    return int(path.stem.split("_")[1])


def get_motor_positions_from_img_paths(paths: List[Path]) -> List[int]:
    """Get all motor positions from a list of image paths

    Parameters
    ----------
    List[Path]
        List of image paths (filenames should be of the format XXX_YYY.png) where XXX is the motor position and YYY is the n'th image taken at that position.

    Returns
    -------
    List[int]
        Motor positions
    """

    return [get_motor_position_from_path(path) for path in paths]


def get_img_step_idx_from_img_paths(paths: List[Path]) -> List[int]:
    """Get all image step indices from a list of image paths

    Parameters
    ----------
    List[Path]
        List of image paths (filenames should be of the format XXX_YYY.png) where XXX is the motor position and YYY is the n'th image taken at that position.

    Returns
    -------
    List[int]
        List of image step indices
    """

    return [get_image_step_idx_from_path(path) for path in paths]


def generate_relative_position_folders(
    save_dir: Path, relative_positions: List[int]
) -> None:
    """Attempt to create relative position folders if they have not been made already.

    Relative position here refers to the relative position from the peak focus. I.e a folder
    -2/ means all images in there are -2 (two steps below) peak focus.

    Parameters
    ----------
    save_dir: Path
        Top level directory where all the relative position subfolders will be saved.

    relative_positions: List[int]
    """

    for pos in relative_positions:
        try:
            Path.mkdir(save_dir / str(pos))
        except FileExistsError:
            pass


def copy_img_to_folder(img_path: Path, save_dir: Path, relative_pos: int) -> None:
    """Copy an image to the appropriate relative position folder (i.e relative position to focus)

    Parameters
    ----------
    img_path: Path
        Path to the image to be copied
    save_dir: Path
        Path to the top-level directory under which all the relative position folders (i.e +30, +29, ..., -29, -30) are stored
    relative_pos: int
        Relative position of the image to the peak focus
    """
    copy(img_path, save_dir / str(relative_pos))


def move_imgs_to_relative_pos_folders(
    img_paths: List[Path], save_dir: Path, relative_positions: List[int]
) -> None:
    """Copy images to their respective relative position folders.

    Parameters
    ----------
    img_paths: List[Path]
    save_dir: Path
        Path to the top-level directory under which all the relative position folders (i.e +30, +29, ..., -29, -30) are stored
    relative_positions: List[int]
        Relative positions of the images to the peak focus
    """

    for img_path, pos in zip(img_paths, relative_positions):
        copy_img_to_folder(img_path, save_dir, pos)


def log_power_spectrum_radial_average_sum(img: np.ndarray) -> float:
    def radial_average(data: np.ndarray) -> np.ndarray:
        data = data / np.max(data)
        h, w = data.shape
        center = (w // 2, h // 2)
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(int)

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    power_spectrum = np.fft.fftshift(np.fft.fft2(img))
    log_ps = np.log(np.abs(power_spectrum))
    return np.sum(radial_average(log_ps))


def multiprocess_focus_metric(
    imgs: List[np.ndarray], metric_fn: Callable[[np.ndarray], float]
) -> List[float]:
    """Get focus metrics out of a set of images, use multiprocessing to speed up computation.

    Parameters
    ----------
    imgs: List[np.ndarray]
    metric_fn: Callable
        A callable function which returns a float value - this is the focus metric that will be run on each image

    Returns
    -------
    List[float]
    """

    with Pool() as pool:
        focus_metrics = list(tqdm(pool.imap(metric_fn, imgs), total=len(imgs)))
    return focus_metrics


def group_by_motor_positions(
    items: List[Any], motor_positions: List[int]
) -> List[List[Any]]:
    """Given a list of items (e.g. images, paths, focus metrics) and motor positions that
    correspond to those items, this groups the items together by their focus metrics

    Parameters
    ----------
    items: List[Any]
        List of items to group together
    motor_positions: List[int]
        List of motor positions corresponding to the items

    Returns
    -------
    Dict[int, List[Any]]
        Dictionary of motor positions to items
    """
    motor_pos_to_items = defaultdict(list)
    for item, pos in zip(items, motor_positions):
        motor_pos_to_items[pos].append(item)
    sorted_motor_positions = sorted(motor_pos_to_items.items())
    return [item for _, item in sorted_motor_positions]


def get_relative_to_peak_positions(
    motor_positions: List[int], peak_pos: int
) -> List[int]:
    """Get the relative position to the peak focus position.

    Parameters
    ----------
    motor_positions: List[int]
        The positions at which the images were taken (stored in the filename of the images)
    peak_pos: int
        The position at which the peak focus was found

    Returns
    -------
    List[int]
    """

    return [motor_pos - peak_pos for motor_pos in motor_positions]
