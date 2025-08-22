from pathlib import Path
import multiprocessing

import numpy as np
from tqdm import tqdm
import zarr


def gradientAverage(img: np.ndarray):
    """Returns the normalized gradient average (in x and y)"""
    gx, gy = np.gradient(img) / np.max(img)
    return np.average(np.sqrt(gx**2 + gy**2))


def get_zf(p):
    p = Path(p)
    zf_path = p / (p.stem + ".zip")
    if zf_path.exists():
        try:
            zf = zarr.open(p / (p.stem + ".zip"), "r")
            return zf
        except:
            return None
    else:
        return None


def get_classic_focus_for_experiment(zf_path: Path):
    """
    Given a zarr file, returns the gradient average focus values for each image in the zarr file.

    Parameters
    ----------
    zf_path : Path

    Returns
    -------
    List[float]
        A list of the focus values for each image in the zarr file.
    """

    zf = zarr.open(zf_path, "r")
    focus_vals = [None] * zf.initialized

    for i in range(zf.initialized):
        img = zf[:, :, i]
        focus_vals[i] = gradientAverage(img)

    return focus_vals


def process_dataset(p):
    np_save_path = (
        Path("/hpc/projects/group.bioengineering/LFM_scope/tororo_classic_focus")
        / f"{Path(p).stem}grad_avgs.npy"
    )
    if np_save_path.exists():
        return
    zf = get_zf(p)
    print(f"Working on {p}")
    if zf is not None:
        gavgs = np.zeros(zf.initialized, dtype=np.float16)
        for i in range(zf.initialized):
            img = zf[:, :, i]
            gavgs[i] = gradientAverage(img)
        np.save(np_save_path, gavgs)


if __name__ == "__main__":
    files = "/home/ilakkiyan.jeyakumar/all_tororo_paths.txt"
    with open(files, "r") as f:
        paths = f.readlines()
        paths = [x.strip() for x in paths]

    # Creating a multiprocessing pool with the default number of processes
    with multiprocessing.Pool(64) as pool:
        results = list(
            tqdm(pool.imap_unordered(process_dataset, paths), total=len(paths))
        )
