from typing import List, Dict
from pathlib import Path
from csv import DictReader


def get_list_of_per_image_metadata_files(top_level_dir: str) -> List[Path]:
    """Get a list of all the per image metadata in this folder and all its subfolders

    Parameters
    ----------
    top_level_dir : str
        Top level directory path

    Returns
    -------
    List[Path]
    """
    return sorted(Path(top_level_dir).glob("**/*perimage*.csv"))


def get_list_of_experiment_level_metadata_files(top_level_dir: str) -> List[Path]:
    """Get a list of all the experiment-levels metadata in this folder and all its subfolders

    Parameters
    ----------
    top_level_dir : str
        Top level directory path

    Returns
    -------
    List[Path]
    """
    return sorted(Path(top_level_dir).glob("**/*exp*.csv"))


def parse_csv(filepath: str) -> Dict:
    """Read the csv file and return a dictionary mapping keys (column headers) to a list of values.

    Parameters
    ----------
    filepath : str
        Path of csv file

    Returns
    -------
    Dict
    """
    d = {}
    with open(filepath) as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            for key in row.keys():
                d.setdefault(key, []).append(row[key])
    return d
