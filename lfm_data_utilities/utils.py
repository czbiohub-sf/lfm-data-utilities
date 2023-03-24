from typing import List, Dict, Tuple
from collections import namedtuple
from pathlib import Path
from csv import DictReader
from datetime import datetime
from multiprocessing import Pool
from zipfile import BadZipFile

import zarr
from tqdm import tqdm


Dataset = namedtuple('Dataset', ['zarr_path', 'per_img_csv_path', 'experiment_csv_path', 'subsample_path'])

def get_full_datasets(top_level_dir: str) -> List[Dataset]:
    """Get a list of all 'full' datasets, i.e all folders which contain a valid zarr file, per image csv file,
    and experiment metadata csv file.

    Returns a list of "Dataset" named tuples. Access is as follows (imagine you pick one dataset, d, out of the list):
        > Zarr file:            d.zarr_path
        > per image metadata:   d.per_img_csv_path
        > experiment metadata:  d.experiment_csv_path
        > subsample directory:  d.subsample_path

    Searches recursively through all subdirectories.
    IMPORTANT NOTE:
        This returns only 'full' datasets - that is, paths to folders which contain:
            1. A valid zarr file
            2. A per image csv file
            3. An experiment metadata csv file
            4. A subsample image directory
        This means folders without all of the above WILL BE EXCLUDED!
        Failed runs typically don't have a subsample directory generated.
        Keep this in mind!

    If you want to exhaustively get all the zarr files, per image metadata files, experiment metadata files,
    or subsample directories, use one of the more specific functions below.

    This function is here as a convenience. Typically calling 'glob' from scratch (i.e the top level directory)
    can be slow.

    Parameters
    ----------
    top_level_dir: str
        Top level directory path to search
    """

    valid_datasets: List[Dataset] = []
    per_img_csv_paths = get_list_of_per_image_metadata_files(top_level_dir)
    for per_img in per_img_csv_paths:
        zfp = get_list_of_zarr_files(per_img.parent)
        efp = get_list_of_experiment_level_metadata_files(per_img.parent)
        ssp = get_list_of_subsample_dirs(per_img.parent)

        if per_img and zfp and efp and ssp:
            valid_datasets.append(Dataset(zfp, per_img, efp, ssp))

    return valid_datasets


def get_list_of_zarr_files(top_level_dir: str) -> List[Path]:
    """Get a list of all the zarr (saved as .zip) files in this folder and all its subfolders

    Parameters
    ----------
    top_level_dir : str
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(Path(top_level_dir).rglob("*.zip"))


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

    return sorted(Path(top_level_dir).rglob("*perimage*.csv"))


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

    return sorted(Path(top_level_dir).rglob("*exp*.csv"))

def get_list_of_subsample_dirs(top_level_dir: str) -> List[Path]:
    """Get a list of all the sub sample image directories

    Parameters
    ----------
    top_level_dir : str
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(Path(top_level_dir).rglob("*sub_sample*/"))


def get_list_of_oracle_run_folders(top_level_dir: str) -> List[Path]:
    """Get a list of all the folders created on each oracle run sorted by date.

    Parameters
    ----------
    top_level_dir: str

    Returns
    -------
    List[Path]
    """

    tlds = [
        x
        for x in Path(top_level_dir).glob("*/")
        if "logs" not in x.stem and not "." in x.stem and Path.is_dir(x)
    ]

    return tlds


def get_dates_from_top_level_folders(tld_folders: List[Path]) -> List[str]:
    return sorted(
        list({}.fromkeys([x.stem.rsplit("-", 1)[0] for x in tld_folders]))
    )  # Remove duplicates, ignore the hour/minute/second part of the timestamp


def get_all_metadata_files_in_date_range(
    metadata_filepaths: List[Path], d1: datetime, d2: datetime
) -> List[Path]:
    """Get all the metadata files taken within the date range [d1, d2]

    Parameters
    ----------
    metadata_filepaths: List[Path]
    d1: datetime
        First date
    d2: datetime
        Second (later) date

    Returns
    -------
    List[Path]
    """

    return [
        f
        for f in metadata_filepaths
        if d1 <= parse_datetime_string(f.parent.parent.stem) <= d2
    ]


def get_all_metadata_files_from_same_day(
    metadata_filepaths: List[Path], d1: datetime
) -> List[Path]:
    """Get all the metadata files that were taken on a particular day.

    Parameters
    ----------
    metadata_filepaths: List[Path]
    d1: datetime

    Returns
    -------
    List[Path]
    """

    return [
        f
        for f in metadata_filepaths
        if parse_datetime_string(f.parent.parent.stem).date() == d1.date()
    ]


def get_date_range_from_user(
    date_format: str = "%Y-%m-%d",
) -> Tuple[datetime, datetime]:
    """Prompt a user for a date range, parse, and return two datetime objects.

    Parameters
    ----------
    date_format: str = "%Y-%m-%d"

    Returns
    -------
    Tuple[datetime, datetime]

    Exceptions
    ----------
    Will raise an exception if there was an error parsing the user input.
    """

    print("You will be prompted to enter two dates on separate lines.")
    print(
        "You can leave either or both of the inputs empty.\n"
        "If you leave the first empty, all dates leading up to and including the second will be used.\n"
        "If you leave the second empty, all dates from the first date to the end will be used.\n"
        "If both are empty, all datasets will be used."
    )
    d1 = input("date 1 (YYYY-MM-DD): ")
    d2 = input("date 2 (YYYY-MM-DD): ")
    try:
        d1 = datetime.min if d1 == "" else datetime.strptime(d1, date_format)
        d2 = datetime.max if d2 == "" else datetime.strptime(d2, date_format)

        return d1, d2
    except Exception as e:
        print(f"There was an error parsing one of the inputs: {e}")
        raise


def parse_datetime_string(filename: str, format: str = "%Y-%m-%d-%H%M%S") -> datetime:
    """Superfluous, Convenience function to parse a datetime string.

    Parameters
    ----------
    filename: str
    format: str="%Y-%m-%d-%H%M%S"

    Returns
    -------
    datetime
    """

    return datetime.strptime(filename, format)


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


def load_read_only_zarr(zarr_path: str) -> zarr.core.Array:
    """Load a zarr file - no protections (i.e exception catching) against bad zip files.
    
    Return
    ------
    zarr.core.Array
        Opened zarr file
    """
    return zarr.open(zarr_path, "r")


def load_csv(filepath: str) -> Dict:
    """Read the csv file and return a dictionary mapping keys (column headers) to a list of values.

    Parameters
    ----------
    filepath : str
        Path of csv file

    Returns
    -------
    Dict
        Nested dictionary - filename to dictionary (the inner dictionary maps csv column headers to lists of values, all of which are strings)
        If the value you are interested in is a float/int, you'll need to loop through the inner dictionary's lists and do the appropriate type casting.
    """

    d = {}
    with open(filepath) as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            for key in row.keys():
                d.setdefault(key, []).append(row[key])

    return {"filepath": filepath, "vals": d}


def load_log_file(filepath: str) -> Dict:
    """Get lines, split by newlines, from the log file

    Parameters
    ----------
    filepath : str
        Path of log file

    Returns
    -------
    Dict
        A dictionary mapping filename to a list of lines from the log file
    """
    lines = []

    with open(filepath) as f:
        lines = f.read().splitlines()

    return {"filepath": filepath, "vals": lines}


def multiprocess_load_files(filepaths: List[Path], fn: callable) -> List:
    """Wraps parse_csv with multiprocessing. Takes a list of filepaths to load.

    Parameters
    ----------
    filepaths: List[str]
    fn: Callable
        TODO: Extend to take an arbitrary amount of parameters

    Returns
    -------
    List[Interior type depends on output of callable]
    """

    with Pool() as pool:
        data = list(tqdm(pool.imap(fn, filepaths), total=len(filepaths)))
    return data


def multiprocess_load_zarr(filepaths: List[Path]) -> List[zarr.core.Array]:
    """Multiprocess load zarr files

    Parameters
    ----------
    filepaths: List[str]

    Returns
    -------
    List[zarr.core.Array]
    """

    return multiprocess_load_files(filepaths, load_read_only_zarr)


def multiprocess_load_csv(filepaths: List[Path]) -> List[Dict]:
    """Multiprocess load csv files

    Parameters
    ----------
    filepaths: List[str]

    Returns
    -------
    List[Dict]
        A list of dictionaries mapping columns to a list of their values.
        The dictionary has two keys:
        (key, val)
            "filepath": str
            "vals": Dict - this inner dictionary is what maps column headers to lists
    """

    return multiprocess_load_files(filepaths, load_csv)


def multiprocess_load_log(filepaths: List[Path]) -> List[Dict]:
    """Multiprocess load log files

    Parameters
    ----------
    filepaths: List[str]

    Returns
    -------
    List[Dict]
        A list of dictionaries:
        (key, val)
            "filepath": str
            "vals": List[str] - this is where the log file lines are stored (separated by newline)
    """

    return multiprocess_load_files(filepaths, load_log_file)


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
