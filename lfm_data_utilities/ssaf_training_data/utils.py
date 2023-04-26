import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from os import listdir
from typing import List, Union
from shutil import copy
from pathlib import Path
from multiprocessing import Pool



PathLike = Union[str, Path]


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


def get_list_of_img_paths_in_folder(folder_path: PathLike) -> List[Path]:
    """Get a list of the filepaths of the tiffs or pngs in this folder.

    The only reason we accommodate tiffs is cause the original SSAF data was collected using tiffs, we have
    since changed to saving pngs.

    Parameters
    ----------
    folder_path: str
        Folder containing zstack images
    Returns
    -------
    List[Path]
    """

    pngs = sorted(Path(folder_path).glob("*.png"))
    tiffs = sorted(Path(folder_path).glob("*.tiff"))

    if len(pngs) == 0 and len(tiffs) > 0:
        return tiffs

    elif len(pngs) > 0 and len(tiffs) == 0:
        return pngs

    else:
        raise ValueError(
            f"For some reason there are both pngs and tiffs in this folder: \npngs: {pngs}\ntiffs: {tiffs}"
        )


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


def load_img(img_path: Path) -> np.ndarray:
    """Load an image using opencv

    Parameters
    ----------
    img_path: Path

    Returns
    -------
    np.ndarray
    """

    return cv2.imread(str(img_path), 0)


def load_imgs(img_paths: List[Path]) -> List[np.ndarray]:
    """Multiprocess load and return images

    Parameters
    ----------
    img_paths: List[Path]
        List of image paths to load

    Returns
    -------
    List[np.ndarray]
    """

    with Pool() as pool:
        imgs = list(tqdm(pool.imap(load_img, img_paths), total=len(img_paths)))
    return imgs


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

    [
        copy_img_to_folder(img_path, save_dir, pos)
        for (img_path, pos) in tqdm(
            zip(img_paths, relative_positions), total=len(img_paths)
        )
    ]


def log_power_spectrum_radial_average_sum(img: np.ndarray) -> float:
    def radial_average(data: np.ndarray) -> np.ndarray:
        data = data / np.max(data)
        h, w = data.shape[0], data.shape[1]
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
    imgs: List[np.ndarray], metric_fn: callable
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


def find_peak_position(
    focus_metrics: List[float],
    motor_positions: List[int],
    imgs_per_step: int = 30,
    local_vicinity: int = 10,
    max_motor_pos: int = 900,
    save_loc: Path = False,
    folder_name: str = None,
) -> int:
    """
    Averages the focus metrics (given the number of images per step), does a simple quadratic fit near the vicinity of the peak,
    and returns the motor position of the peak focus.

    Parameters
    ----------
    focus_metrics: List[float]
    motor_positions: List[int]
    imgs_per_step: int=30
    local_vicinity: int=10
        The number of steps (+/-) of the peak focus position about which the quadratic fit will be done
    save: bool=False
        Whether to save the plot of the focus metric and quadratic fit

    Returns
    -------
    int
        Motor position at which the peak focus was found.
    """

    # Get mean and normalize focus metric
    metrics_averaged = np.mean(
        np.asarray(focus_metrics).reshape(-1, imgs_per_step), axis=1
    )
    metrics_normed = metrics_averaged / np.max(metrics_averaged)

    # Get metrics and motor positions for the values in the local vicinity of the peak focus
    motor_pos_nodup = np.unique(motor_positions)
    peak_focus_pos = np.argmax(metrics_normed)
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

    if save_loc:
        plt.figure(figsize=(10, 7))
        plt.plot(
            motor_pos_nodup, metrics_normed, label="Focus metric vs. motor position"
        )
        plt.plot(motor_pos_local_vicinity, curve, label="Curve fit")
        plt.plot(
            peak_focus_motor_position,
            np.max(curve),
            "*",
            label=f"Max@{peak_focus_motor_position}",
        )

        plt.title(f"{folder_name}")
        plt.xlabel("Motor position (steps)")
        plt.ylabel("Focus metric (dimensionless)")
        plt.legend()
        plt.savefig(f"{save_loc / folder_name}.png")

    return peak_focus_motor_position


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
