"""
Point to some training data folders (i.e a folder which contains folders of ..., +3, +2, +1, 0 -1, -2, -3, ... )
This tool will create a video for each of those folders that you can then scrub through to try to spot outliers.
"""

import sys
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm
import cv2

from lfm_data_utilities.utils import get_list_of_img_paths_in_folder, load_imgs_threaded


def make_video_from_pngs(folder_path: Path, save_dir: Path, framerate=30):
    """Generate a video (mp4) from a folder of pngs.

    Parameters
    ----------
    folder_path: Path of folder of pngs
    save_dir: Path to save video
    """

    img_paths = get_list_of_img_paths_in_folder(folder_path)
    imgs = load_imgs_threaded(img_paths)
    height, width = imgs[0].shape
    output_path = Path(save_dir) / Path(folder_path.stem + ".mp4")

    writer = cv2.VideoWriter(
        f"{output_path}",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=framerate,
        frameSize=(width, height),
        isColor=False,
    )

    for path, img in zip(img_paths, imgs):
        filename = path.name
        img = cv2.putText(img, filename, bottomLeftOrigin=(height, 0))
        writer.write(img)
    writer.release()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <path folders> <path to save videos (note: folder will be created if it doesn't exist already)>"
        )
        sys.exit(1)

    path_to_runset = Path(sys.argv[1])
    path_to_save = Path(sys.argv[2])
    folders = [f for f in path_to_runset.glob("*/") if f.name[0] != "."]

    print("Generating videos...")
    with mp.Pool() as pool:
        pool.starmap(make_video_from_pngs, [(x, path_to_save) for x in tqdm(folders)])
