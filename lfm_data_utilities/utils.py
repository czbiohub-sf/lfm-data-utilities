import cv2
import time
import zarr
import traceback
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from csv import DictReader
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Any, Callable, Union


PathLike = Union[str, Path]


@dataclass
class DatasetPaths:
    zarr_path: Path
    per_img_csv_path: Path
    experiment_csv_path: Path
    subsample_path: Path

    @property
    def root_dir(self) -> Path:
        if (
            not self.zarr_path.parent
            == self.per_img_csv_path.parent
            == self.experiment_csv_path.parent
        ):
            raise ValueError(
                f"invalid results for dataset paths - data are from different dirs for {self.zarr_path}"
            )
        return self.zarr_path.parent


class Dataset:
    def __init__(self, dp: DatasetPaths, fail_silently=True):
        self.dp: DatasetPaths = dp
        try:
            self.zarr_file = load_read_only_zarr(str(dp.zarr_path))
            self.per_img_metadata = load_per_img_csv(dp.per_img_csv_path)
            self.experiment_metadata = load_csv(dp.experiment_csv_path)
        except Exception as e:
            print(f"Error loading dataset {dp.zarr_path}: {e}")
            if not fail_silently:
                raise e
            else:
                self.successfully_loaded = False
        else:
            self.successfully_loaded = True


@contextmanager
def timing_context_manager(description: str, precision: int = 5):
    """Context manager for timing code execution.

    Args:
        description (str): description of code to be timed
    """
    # TODO 'quiet' environment variable?
    try:
        start_time = time.perf_counter()
        print(f"{description}...", end=" ", flush=True)
        yield
    finally:
        end_time = time.perf_counter()
        print(f"{end_time - start_time:.{precision}f} s")


def make_video(dataset: Dataset, save_dir: PathLike):
    zf = dataset.zarr_file
    per_img_csv = dataset.per_img_metadata

    # Get duration in seconds
    start = float(per_img_csv["vals"]["timestamp"][0])
    end = float(per_img_csv["vals"]["timestamp"][-1])
    duration = end - start
    num_frames = zf.initialized
    framerate = num_frames / duration
    height, width = zf[:, :, 0].shape

    save_dir.mkdir(exist_ok=True)
    output_path = Path(save_dir) / Path(dataset.dp.zarr_path.stem + ".mp4")

    writer = cv2.VideoWriter(
        f"{output_path}",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=framerate,
        frameSize=(width, height),
        isColor=False,
    )

    for i, _ in enumerate(tqdm(range(num_frames))):
        img = zf[:, :, i]
        writer.write(img)
    writer.release()


def is_not_hidden_path(path: PathLike) -> bool:
    """Check if file is valid or if it's an unreadable temporary file. (Apple saves temporary files as '._<filename>'.)"""

    return not Path(path).name.startswith(".")


def load_datasets(top_level_dir: PathLike, fail_silently: bool=False) -> List[Dataset]:
    """Load all zarr and metadata files. Returns all data in a list of Dataset objects."""

    print(
        "Getting dataset paths (i.e paths to zarr files, per image/experiment level metadata csvs...)"
    )
    print(
        "NOTE: Traversing ess file tree if you are not on Bruno is excruciatingly slow for some reason."
    )
    all_dataset_paths = get_all_dataset_paths(top_level_dir)

    print(
        "Generating dataset objects. Note: Check that a dataset is valid by checking its `successfully_loaded` attribute..."
    )
    return [Dataset(dp, fail_silently=fail_silently) for dp in tqdm(all_dataset_paths)]


def get_all_dataset_paths(
    top_level_dir: PathLike, verbose: bool = False
) -> List[DatasetPaths]:
    """Get a list of all dataset paths. This function will find a list of per image metadata csvs, and then attempt to get the
    zarr, experiment-level metadata file, and subsample directory located in that same folder. If one or more of those are
    not present, the "Dataset" named tuple will have "None" for those parameters.

    Returns a list of "Dataset" dataclass. Access is as follows (imagine you pick one dataset, d, out of the list):
        > Zarr file:            d.zarr_path (might be None)
        > per image metadata:   d.per_img_csv_path
        > experiment metadata:  d.experiment_csv_path (might be None)
        > subsample directory:  d.subsample_path (might be None)

    Searches recursively through all subdirectories.

    This function is here as a convenience. Typically calling 'glob' from scratch (i.e the top level directory)
    for each file type of interest (zarr, per image metadata, experiment metadata, and subsample directory) can be slow.

    Parameters
    ----------
    top_level_dir: str
        Top level directory path to search
    """

    def get_path_or_none(paths: List[PathLike]) -> Optional[PathLike]:
        if len(paths) == 0:
            return None
        elif len(paths) == 1:
            return paths[0]
        raise ValueError(f"more than one possible path: {paths}")

    dataset_paths: List[DatasetPaths] = []
    per_img_csv_paths = get_list_of_per_image_metadata_files(top_level_dir)
    for per_img in per_img_csv_paths:
        zfp = get_path_or_none(get_list_of_zarr_files(per_img.parent))
        efp = get_path_or_none(
            get_list_of_experiment_level_metadata_files(per_img.parent)
        )
        ssp = get_path_or_none(get_list_of_subsample_dirs(per_img.parent))
        verbose_names = [
            ("zarr file", zfp),
            ("experiment metadata", efp),
            ("subsample directory", ssp),
        ]
        if zfp and efp and ssp:
            dataset_paths.append(DatasetPaths(zfp, per_img, efp, ssp))
        else:
            if verbose:
                print(
                    f"missing files in {per_img.parent}: "
                    f"{', '.join(v[0] for v in verbose_names if v[1] is None)}"
                )

    return dataset_paths


def get_list_of_txt_files(
    zarr_files: List[PathLike], top_level_txt_dir: PathLike, suffix: str
) -> List[Optional[Path]]:
    """Get a list of paths to the corresponding .txt files for a given list of zarr files"""

    txt_files = [
        get_corresponding_txt_file(zarr_file, top_level_txt_dir, suffix)
        for zarr_file in zarr_files
    ]
    return [txt_file for txt_file in txt_files if txt_file.exists()]


def get_corresponding_txt_file(
    zarr_file: Path, top_level_txt_dir: PathLike, suffix: str
) -> Optional[Path]:
    """Get the path to the corresponding .txt file for a given zarr file"""

    basename = zarr_file.stem
    txt_file = Path(top_level_txt_dir) / f"{basename}__{suffix}.txt"

    return txt_file


def get_list_of_zarr_files(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all the zarr (saved as .zip) files in this folder and all its subfolders

    Parameters
    ----------
    top_level_dir : PathLike
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(
        [
            file
            for file in Path(top_level_dir).rglob("*.zip")
            if is_not_hidden_path(file)
        ]
    )


def get_list_of_per_image_metadata_files(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all the per image metadata in this folder and all its subfolders

    Parameters
    ----------
    top_level_dir : PathLike
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(
        [
            x
            for x in Path(top_level_dir).rglob("*perimage*.csv")
            if is_not_hidden_path(x)
        ]
    )


def get_list_of_experiment_level_metadata_files(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all the experiment-levels metadata in this folder and all its subfolders

    Parameters
    ----------
    top_level_dir : str
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(
        [
            file
            for file in Path(top_level_dir).rglob("*exp*.csv")
            if is_not_hidden_path(file)
        ]
    )


def get_list_of_subsample_dirs(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all the sub sample image directories

    Parameters
    ----------
    top_level_dir : PathLike
        Top level directory path

    Returns
    -------
    List[Path]
    """
    return sorted(
        p for p in Path(top_level_dir).rglob("*sub_sample*/") if is_not_hidden_path(p)
    )


def get_list_of_oracle_run_folders(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all the folders created on each oracle run sorted by date.

    Parameters
    ----------
    top_level_dir: PathLike

    Returns
    -------
    List[Path]
    """

    tlds = [
        x
        for x in Path(top_level_dir).glob("*/")
        if "logs" not in x.stem and "." not in x.stem and Path.is_dir(x)
    ]

    return tlds


def get_dates_from_top_level_folders(tld_folders: List[PathLike]) -> List[str]:
    return sorted(
        list({}.fromkeys([x.stem.rsplit("-", 1)[0] for x in tld_folders]))
    )  # Remove duplicates, ignore the hour/minute/second part of the timestamp


def get_all_metadata_files_in_date_range(
    metadata_filepaths: List[PathLike], d1: datetime, d2: datetime
) -> List[Path]:
    """Get all the metadata files taken within the date range [d1, d2]

    Parameters
    ----------
    metadata_filepaths: List[PathLike]
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
        if d1 <= parse_datetime_string(Path(f).parent.parent.stem) <= d2
    ]


def get_all_metadata_files_from_same_day(
    metadata_filepaths: List[PathLike], d1: datetime
) -> List[Path]:
    """Get all the metadata files that were taken on a particular day.

    Parameters
    ----------
    metadata_filepaths: List[PathLike]
    d1: datetime

    Returns
    -------
    List[Path]
    """

    return [
        f
        for f in metadata_filepaths
        if parse_datetime_string(Path(f).parent.parent.stem).date() == d1.date()
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


def get_list_of_log_files(top_level_dir: PathLike) -> List[Path]:
    """Get a list of all the logs in this folder

    Parameters
    ----------
    top_level_dir : PathLike
        Top level directory path

    Returns
    -------
    List[Path]
    """

    return sorted(Path(top_level_dir).glob("**/*.log"))


def load_read_only_zarr(zarr_path: PathLike) -> zarr.core.Array:
    """Load a zarr file - no protections (i.e exception catching) against bad zip files.

    Return
    ------
    zarr.core.Array
        Opened zarr file
    """

    return zarr.open(zarr_path, "r")


def load_csv(filepath: PathLike) -> Dict[str, Union[PathLike, Dict[str, List[str]]]]:
    """Read the csv file and return a dictionary mapping keys (column headers) to a list of values.

    Parameters
    ----------
    filepath : PathLike
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


def is_float_like(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False


def load_per_img_csv(filepath: PathLike) -> Dict:
    """
    This loads and standardizies per-image csv files

    issues:
        - `timestamp` can be either a 'seconds from epoch' format or
        the format of '2022-12-13-111345_485858' 'YYYY-MM-DD-HHMMSS_ffffff'
    """
    per_img_csv_raw = load_csv(filepath)

    # fix timestamps
    for i in range(len(per_img_csv_raw["vals"]["timestamp"])):
        if is_float_like(per_img_csv_raw["vals"]["timestamp"][i]):
            # assume that timestamp is seconds from epoch for all,
            # so leave it all alone
            break

        per_img_csv_raw["vals"]["timestamp"][i] = datetime.strptime(
            per_img_csv_raw["vals"]["timestamp"][i],
            "%Y-%m-%d-%H%M%S_%f",
        ).timestamp()

    return per_img_csv_raw


def load_log_file(filepath: PathLike) -> Dict:
    """Get lines, split by newlines, from the log file

    Parameters
    ----------
    filepath : PathLike
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


print_lock = mp.Lock()


def protected_fcn(f, *args):
    try:
        f(*args)
    except:
        with print_lock:
            print(f"exception occurred processing {args}")
            print(traceback.format_exc())


def multiprocess_fn(
    argument_list: List[Any],
    fn: Callable[
        [
            Any,
        ],
        Any,
    ],
    ordered: bool = True,
    verbose: bool = True,
) -> List[Any]:
    """Wraps any function invocation in multiprocessing, with optional TQDM for progress.

    Takes a list of arguments for fn, which takes one input. Note that you can use
    functools.partial to fill in any other arguments.

    Parameters
    ----------
    argument_list: List[Any]
    fn: Callable[[Any,], Any]
        TODO: Extend to take an arbitrary amount of parameters
    ordered: bool=True
        return results ordered if true. If false, use imap_unordered which may give a performance boost
    verbose: bool=True
        Show progress bar if true

    Returns
    -------
    List[Interior type depends on output of callable]
    """
    protected_fcn_partial = partial(protected_fcn, fn)

    with mp.Pool() as pool:
        if ordered:
            mp_func = pool.imap
        else:
            mp_func = pool.imap_unordered
        return list(
            tqdm(
                mp_func(protected_fcn_partial, argument_list),
                total=len(argument_list),
                disable=not verbose,
            )
        )


def multiprocess_load_zarr(filepaths: List[PathLike]) -> List[zarr.core.Array]:
    """Multiprocess load zarr files

    Parameters
    ----------
    filepaths: List[PathLike]

    Returns
    -------
    List[zarr.core.Array]
    """

    return multiprocess_fn(filepaths, load_read_only_zarr)


def multiprocess_load_csv(filepaths: List[PathLike]) -> List[Dict]:
    """Multiprocess load csv files

    Parameters
    ----------
    filepaths: List[PathLike]

    Returns
    -------
    List[Dict]
        A list of dictionaries mapping columns to a list of their values.
        The dictionary has two keys:
        (key, val)
            "filepath": str
            "vals": Dict - this inner dictionary is what maps column headers to lists
    """

    return multiprocess_fn(filepaths, load_csv)


def multiprocess_load_log(filepaths: List[PathLike]) -> List[Dict]:
    """Multiprocess load log files

    Parameters
    ----------
    filepaths: List[PathLike]

    Returns
    -------
    List[Dict]
        A list of dictionaries:
        (key, val)
            "filepath": str
            "vals": List[str] - this is where the log file lines are stored (separated by newline)
    """

    return multiprocess_fn(filepaths, load_log_file)


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
