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

    thumbnail_dir: Path = args.path_to_thumbnail_dir
    output_dir: Path = args.path_to_output_dir

    # Get all the corrected and complete folders
    if not args.path_to_thumbnail_dir.exists():
        raise ValueError(
            f"Thumbnail directory {args.path_to_thumbnail_dir} does not exist"
        )

    # Create the folder where the thumbnails will be copied
    output_dir.mkdir(exist_ok=True)

    print(
        "\nFinding all the completed and corrected_* folders (we'll skip the normal healthy dirs, but we'll keep the corrected_healthy folders)"
    )
    completed_dirs = [
        d
        for d in thumbnail_dir.rglob("*completed*")
        if d.stem
        in [
            "ring",
            "trophozoite",
            "schizont",
            "gametocyte",
            "wbc",
            "misc",
        ]
    ]

    corrected_dirs = [
        x
        for x in thumbnail_dir.rglob("corrected_*")
        if x.stem
        in [
            "corrected_healthy",
            "corrected_ring",
            "corrected_trophozoite",
            "corrected_schizont",
            "corrected_gametocyte",
            "corrected_wbc",
            "corrected_misc",
        ]
    ]

    # Remove the healthy dirs (how we'll keep the corrected_healthy dirs)
    completed_dirs = [dir for dir in completed_dirs if "healthy" not in dir.stem]

    combined = completed_dirs + corrected_dirs

    # Get all the thumbnails from the corrected folders
    for thumbnail_dir in tqdm(
        combined, desc="Looping through completed/corrected folders"
    ):
        for thumbnail in thumbnail_dir.rglob("*.png"):
            verified_class = get_verified_class_from_thumbnail_path(thumbnail)
            if "corrected" in verified_class:
                verified_class = verified_class.split("_")[1]
            output_dir = args.path_to_output_dir / verified_class
            output_dir.mkdir(exist_ok=True)
            shutil.copy(thumbnail, output_dir)
