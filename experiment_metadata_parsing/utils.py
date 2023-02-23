from typing import List
from pathlib import Path
from csv import DictReader


def get_list_of_per_image_metadata_files(top_level_dir: str) -> List[Path]:
    """Get a list of all the per image metadata in this folder and all its subfolders

    Returns
    -------
    List[Path]
    """
    return sorted(Path(top_level_dir).glob("**/*perimage*.csv"))


def get_list_of_experiment_level_metadata_files(top_level_dir: str) -> List[Path]:
    """Get a list of all the experiment-levels metadata in this folder and all its subfolders

    Returns
    -------
    List[Path]
    """
    return sorted(Path(top_level_dir).glob("**/*exp*.csv"))
