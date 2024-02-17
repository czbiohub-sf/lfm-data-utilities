import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import cv2
import numpy as np

def filter_for_conf(arr, conf_thresh):
    return arr[:, arr[7, :] > conf_thresh]

def parse_filepaths(filepath):
    if filepath.endswith('.txt'):
        with open(filepath, 'r') as file:
            lines = file.readlines()
            filepaths = [Path(line.strip()) for line in lines]
    elif filepath.endswith('.npy'):
        filepaths = [Path(filepath)]
    else:
        raise ValueError("Invalid file format. Only .txt and .npy files are supported.")
    
    return filepaths

def get_imgs_dir(npy_file: Path) -> Path:
    return npy_file.parent.parent.parent / "images"

def get_thumbnail_coords(npy_filepath: Path) -> List[Tuple[Path, Tuple[int, int, int, int]]]:
    """
    Get the corresponding image file path and 
    coordinates of the parasitic cells in the .npy file.
    """
    imgs_dir = get_imgs_dir(npy_filepath)

    # Load and filter npy file
    arr = np.load(npy_filepath)
    arr = filter_for_conf(arr, 0.9)
    # widths = arr[3, :] - arr[1, :]
    # heights = arr[4, :] - arr[2, :]
    # ar = widths / heights

    # ar_mask = np.logical_and(0.8 <= ar, ar <= 1.2)

    # arr = arr[:, ar_mask]

    classes = np.unique(arr[6, :])

    class_paths_and_coords = []
    for c in classes:
        # Skip healthy
        if not c in [1, 2, 3, 4]:
            continue

        class_filter = arr[:, arr[6, :] == c]
        for i in range(class_filter.shape[1]):
            img_index = int(class_filter[0, i])
            img_path = imgs_dir / f"img_{img_index:05d}.png"
            tlx, tly, brx, bry = [int(x) for x in class_filter[1:5, i]]

            class_paths_and_coords.append((int(c), img_path, (tlx, tly, brx, bry)))
    
    return class_paths_and_coords

def crop_and_save_thumbnail(path_and_coords: Tuple[int, Path, int, int, int, int], output_dir: Path):
    yogo_class, img_path, (tlx, tly, brx, bry) = path_and_coords
    img = cv2.imread(str(img_path))
    cropped_img = img[tly:bry, tlx:brx]
    output_path = output_dir / (f"{yogo_class}" + f"_{img_path.stem}.png")

    try:
        cv2.imwrite(str(output_path), cropped_img)
    except Exception:
        # Dimensions of the thumbnail are 0
        pass

def main():
    parser = argparse.ArgumentParser(description='Process filepaths.')
    parser.add_argument('filepath', type=str, help='Path to a .npy file or a .txt file containing filepaths')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    
    filepaths = parse_filepaths(args.filepath)

    paths_and_coords = [get_thumbnail_coords(x) for x in tqdm(filepaths, desc="Loading files")]
    paths_and_coords = [item for sublist in paths_and_coords for item in sublist]

    partial_crop_and_save_thumbnail = partial(crop_and_save_thumbnail, output_dir=Path(args.output_dir))
    
    with Pool() as pool:
        list(tqdm(pool.imap(partial_crop_and_save_thumbnail, paths_and_coords), desc="Saving thumbnails", total=len(paths_and_coords)))

if __name__ == '__main__':
    main()