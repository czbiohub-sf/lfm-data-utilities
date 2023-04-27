#! /usr/bin/env python3

""" Jobs for this file:

1 find all run folders (by zarr file)
2 run SSAF, YOGO, and flowrate on all images
3 save results in a csv file
4 save metadata in a yml file
"""

import types
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

from pathlib import Path
from typing import Optional
from functools import partial

import git
import yogo
import torch
import autofocus

from tqdm import tqdm
from ruamel.yaml import YAML

from lfm_data_utilities import utils
from lfm_data_utilities.image_processing.flowrate_utils import (
    get_all_flowrates_from_experiment,
)


def try_get_package_version_identifier(package: types.ModuleType) -> Optional[str]:
    """
    Try to get the git commit hash of the package, if it exists.
    If it doesn't, return the __version__. If that doesnt exist, return None.
    """
    try:
        repo = git.Repo(package.__path__[0], search_parent_directories=True)
        return repo.head.commit.hexsha
    except AttributeError:
        try:
            return package.__version__
        except AttributeError:
            return None


def write_metadata_for_dataset_path(
    output_parent_dir: Path,
    autofocus_pth_path: Path,
    yogo_pth_path: Path,
):
    autofocus_package_id = try_get_package_version_identifier(autofocus)
    yogo_package_id = try_get_package_version_identifier(yogo)
    # write all the above to meta.yml in output_parent_dir
    yaml = YAML(typ="safe")
    meta = {
        "autofocus_package_id": autofocus_package_id,
        "yogo_package_id": yogo_package_id,
        "autofocus_pth_path": autofocus_pth_path,
        "yogo_pth_path": yogo_pth_path,
    }
    with open(output_parent_dir / "meta.yml", "w") as f:
        yaml.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create dense data from runs for vetting")
    parser.add_argument("path_to_runset", type=Path, help="path to run folders")
    parser.add_argument(
        "output_dir",
        type=Path,
        default=None,
        help="path to output directory - defaults to path_to_runset/../output",
    )
    parser.add_argument(
        "path_to_yogo_pth", type=Path, default=None, help="path to yogo pth file"
    )
    parser.add_argument(
        "path_to_autofocus_pth",
        type=Path,
        default=None,
        help="path to autofocus pth file",
    )

    args = parser.parse_args()

    if not args.path_to_runset.exists() or not args.path_to_runset.is_dir():
        raise ValueError(f"{args.path_to_runset} does not exist or is not a dir")
    elif args.output_dir.exists() and (not args.output_dir.is_dir()):
        raise ValueError(f"{args.output_dir} is not a dir")
    elif not args.path_to_yogo_pth.exists():
        raise ValueError(f"{args.path_to_yogo_pth} does not exist")
    elif not args.path_to_autofocus_pth.exists():
        raise ValueError(f"{args.path_to_autofocus_pth} does not exist")

    # args.output_dir.mkdir(exist_ok=True)

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) != 2:
        # TODO generalize for 1 or 0 devices
        raise ValueError(f"need exactly two cuda devices, got {len(devices)}")

    with utils.timing_context_manager("getting all dataset paths"):
        dataset_paths = utils.get_all_dataset_paths(args.path_to_runset, verbose=False)

    print(f"starting calculation on {len(dataset_paths)} datasets")
    for dataset_path in tqdm(dataset_paths):
        pool = ProcessPoolExecutor(max_workers=3)
        flowrate_job = partial(
            get_all_flowrates_from_experiment,
            dataset_path=dataset_path,
            verbose=False,
        )
        autofocus_job = partial(
            autofocus.predict,
            pth_path=args.path_to_autofocus_pth,
            path_to_zarr=utils.load_read_only_zarr(dataset_path.zarr_path),
            device=devices[0],
        )
        yogo_job = partial(
            yogo.predict,
            pth_path=args.path_to_yogo_pth,
            path_to_zarr=utils.load_read_only_zarr(dataset_path.zarr_path),
            device=devices[1],
        )

        flowrate_future = pool.submit(flowrate_job)
        autofocus_future = pool.submit(autofocus_job)
        yogo_future = pool.submit(yogo_job)

        wait(
            [flowrate_future, autofocus_future, yogo_future], return_when=ALL_COMPLETED
        )

        print(
            f"flowrate: {flowrate_future.result()}\n"
            f"autofocus: {autofocus_future.result()}\n"
            f"yogo: {yogo_future.result()}\n"
        )
