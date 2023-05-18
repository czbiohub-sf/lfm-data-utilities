"""
Point to some training data folders (i.e a folder which contains folders of ..., +3, +2, +1, 0 -1, -2, -3, ... )
This tool will create a video for each of those folders that you can then scrub through to try to spot outliers.
"""

import sys
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from lfm_data_utilities.utils import make_video_from_pngs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <path folders> <path to save videos (note: folder will be created if it doesn't exist already)>"
        )
        sys.exit(1)

    path_to_runset = sys.argv[1]
    path_to_save = Path(sys.argv[2])
    folders = [f for f in path_to_runset.glob("*/") if f.name[0] != "."]

    print("Generating videos...")
    with mp.Pool() as pool:
        pool.starmap(make_video_from_pngs, [(x, path_to_save) for x in tqdm(folders)])
