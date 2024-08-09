"""

Description: 

This script is used to go through LFM_scope/thumbnail-corrections
and pulled out all the thumbnails of each class which are either in 
correcteD_* (e.g corrected_healthy, corrected_ring, etc.) or in their original
folders but which have which have been completed (i.e healthy/0-completed-paul for example).

Author: Ilakkiyan Jeyakumar
Date: 2024-08-06
"""

import argparse
from pathlib import Path
import shutil

from tqdm import tqdm

from utils import get_verified_class_from_thumbnail_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_thumbnail_dir",
        type=Path,
        help="Path to the top level directory containing all the thumbnails (i.e thumbnail-corrections)",
    )
    parser.add_argument(
        "path_to_output_dir",
        type=Path,
        help="Path to where to save all the thumbnails (this will create the folder and the necessary subfolders)",
    )
    args = parser.parse_args()

    # Get all the corrected and complete folders
    if not args.path_to_thumbnail_dir.exists():
        raise ValueError(
            f"Thumbnail directory {args.path_to_thumbnail_dir} does not exist"
        )

    # Create the folder where the thumbnails will be copied

    thumbnail_dir: Path = args.path_to_thumbnail_dir
    completed_dirs = list(thumbnail_dir.rglob("*completed*"))
    corrected_dirs = list(thumbnail_dir.rglob("corrected_*"))

    # Get all the thumbnails from the corrected folders
    for dir in tqdm(
        [completed_dirs, corrected_dirs],
        desc="Looping through completed/corrected folders",
    ):
        for thumbnail_dir in dir:
            for thumbnail in tqdm(
                thumbnail_dir.iterdir(),
                desc=f"Looping through thumbnails in: {thumbnail_dir}",
            ):
                verified_class = get_verified_class_from_thumbnail_path(thumbnail)
                output_dir = args.path_to_output_dir / verified_class
                output_dir.mkdir(exist_ok=True)
                shutil.copy(thumbnail, output_dir)
