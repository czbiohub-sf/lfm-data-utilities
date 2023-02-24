from typing import List, Dict
from pathlib import Path
from csv import DictReader
from multiprocessing import Pool

from tqdm import tqdm


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


def get_list_of_log_files(top_level_dir: str) -> List[Path]:
    """Get a list of all the logs in this folder

    Parameters
    ----------
    top_level_dir : str
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(Path(top_level_dir).glob("**/*.log"))


def load_csv(filepath: str) -> Dict:
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


def multiprocess_load_files(filepaths: List[Path], fn: callable) -> List:
    """Wraps parse_csv with multiprocessing. Takes a list of filepaths to load.

    Parameters
    ----------
    filepaths: List[str]

    Returns
    -------
    List[Interior type depends on output of callable]
    """

    with Pool() as pool:
        data = list(tqdm(pool.imap(fn, filepaths), total=len(filepaths)))
    return data


def multiprocess_load_csv(filepaths: List[Path]) -> List[Dict]:
    """Multiprocess load csv files

    Parameters
    ----------
    filepaths: List[str]

    Returns
    -------
    List[Dict]
        A list of dictionaries mapping columns to a list of their values
    """

    return multiprocess_load_files(filepaths, load_csv)


def multiprocess_load_log(filepaths: List[Path]) -> List[List[str]]:
    """Multiprocess load log files

    Parameters
    ----------
    filepaths: List[str]

    Returns
    -------
    List[List[str]]
        For each file, a list of strings (lines) from that file, separated by newline.
    """

    return multiprocess_load_files(filepaths, load_log_file)


def load_log_file(filepath: str) -> List[str]:
    """Get lines, split by newlines, from the log file

    Parameters
    ----------
    filepath : str
        Path of log file

    Returns
    -------
    List[str]
        List of lines from the log file
    """

    lines = []

    with open(filepath) as f:
        lines = f.read().splitlines()

    return lines


def get_autobrightness_vals_from_log(lines: List[str]) -> List[float]:
    """Parse through log file lines and return a list of autobrightness values

    Parameters
    ----------
    lines: List[str]
        Lines extracted from the log file (i.e run `load_log_file` and pass in the output of that function)

    Returns
    -------
    List[float]
    """

    # Get relevant lines
    autobrightness_vals = [l for l in lines if "Mean pixel val" in l]

    # Parse relevant lines and get the autobrightness value out
    autobrightness_vals = [l[: l.find("[")].strip() for l in autobrightness_vals]
    autobrightness_vals = [float(l.split(" ")[-1][:-1]) for l in autobrightness_vals]

    return autobrightness_vals
