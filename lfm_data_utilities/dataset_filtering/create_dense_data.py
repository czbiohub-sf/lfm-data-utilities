#! /usr/bin/env python3

""" Jobs for this file:

1 find all run folders (by zarr file)
2 run SSAF, YOGO, and flowrate on all images
3 save metadata in a yml file
4 save results in a csv file
"""

import csv
import types
import argparse
import warnings
import traceback

from pathlib import Path
from itertools import cycle, chain
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import git
import torch

# import these so we can get the package version identifiers (commit or __version__)
import yogo
import autofocus

from yogo import infer as yogo_infer
from autofocus import infer as autofocus_infer

from tqdm import tqdm
from ruamel.yaml import YAML

from lfm_data_utilities import utils
from lfm_data_utilities.malaria_labelling.labelling_constants import CLASSES
from lfm_data_utilities.image_processing.flowrate_utils import (
    get_all_flowrates_from_experiment,
)


def write_metadata_for_dataset_path(
    output_dir: Path,
    autofocus_path_to_pth: Path,
    yogo_path_to_pth: Path,
):
    autofocus_package_id = utils.try_get_package_version_identifier(autofocus)
    yogo_package_id = utils.try_get_package_version_identifier(yogo)
    # write all the above to meta.yml in output_dir
    yaml = YAML()
    meta = {
        "autofocus_package_id": autofocus_package_id,
        "yogo_package_id": yogo_package_id,
        "autofocus_path_to_pth": str(autofocus_path_to_pth.absolute()),
        "yogo_path_to_pth": str(yogo_path_to_pth.absolute()),
    }
    with open(output_dir / "meta.yml", "w") as f:
        yaml.dump(meta, f)


def calculate_yogo_summary(
    pred: torch.Tensor,
    threshold_class_probabilities: bool = True,
    objectness_threshold: float = 0.5,
) -> List[float]:
    """calculate class summary predictions for pred

    if t0_i is the objectness of grid cell i and C_ij is the probability
    that grid cell i has class j, then we return

        if threshold_class_probabilities:
            sum over all grid cells i (argmax(C_ij) | t0_i > 0.5)
        else:
            sum over all grid cells i (C_ij | t0_i > 0.5)

    threshold_class_probabilities=True will give the class with the highest probability for each grid cell,
    which we use in practice. threshold_class_probabilities=False will give expected number of cells of
    each class given t0_i > 0.5.
    """
    pd, Sy, Sx = pred.shape
    num_classes = pd - 5
    reformatted = pred.reshape(pd, Sy * Sx).T
    objectness_threshold_mask = reformatted[:, 4] > objectness_threshold
    predicted_cells = reformatted[objectness_threshold_mask, 5:]

    if threshold_class_probabilities:
        result = (
            torch.nn.functional.one_hot(
                torch.argmax(predicted_cells, dim=1), num_classes=num_classes
            )
            .sum(dim=0)
            .float()
        )
    else:
        result = predicted_cells.sum(dim=0)

    return result.tolist()


def write_results(
    output_dir: Path,
    flowrate_results: Tuple[List[float]],
    autofocus_results: Tuple[List[float]],
    yogo_results: Tuple[torch.Tensor],
    threshold_class_probabilities: bool = True,
):
    """Write results to csv file

    columns are:
        img_idx flowrate_dx flowrate_dy flowrate_confidence focus *calculated_yogo_summary
    """
    # flowrate of 1st frame can't be calculated, so set to 0
    flowrate_iterable = chain(((0, 0, 0),), zip(*flowrate_results))
    with open(output_dir / "data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "img_idx",
                "flowrate_dx",
                "flowrate_dy",
                "flowrate_confidence",
                "autofocus",
                *CLASSES,
            ]
        )
        for i, results in enumerate(
            zip(flowrate_iterable, autofocus_results, yogo_results)
        ):
            (
                flowrate_results,
                autofocus_res,
                yogo_res,
            ) = results
            (
                flowrate_dx,
                flowrate_dy,
                flowrate_confidence,
            ) = flowrate_results
            writer.writerow(
                [
                    str(r)
                    for r in [
                        i,
                        flowrate_dx,
                        flowrate_dy,
                        flowrate_confidence,
                        autofocus_res,
                        *calculate_yogo_summary(
                            yogo_res,
                            threshold_class_probabilities=threshold_class_probabilities,
                        ),
                    ]
                ]
            )


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
    parser.add_argument(
        "--expected-class-probabilities",
        action="store_true",
        default=False,
        help=(
            "if set, calculate expected number of cells per class instead of the sum "
            "of argmax of class probabilities (default)"
        ),
    )
    parser.add_argument(
        "--objectness-threshold",
        type=float,
        default=0.5,
        help="threshold for objectness - class statistics are calculated given this threshold is met (default 0.5)",
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

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        warnings.warn("no cuda devices found, using cpu - this will be slow!")
        devices = cycle(["cpu"])
    else:
        devices = cycle([f"cuda:{i}" for i in range(num_gpus)])

    with utils.timing_context_manager("getting all dataset paths"):
        dataset_paths = utils.get_all_dataset_paths(args.path_to_runset, verbose=False)

    print(f"starting calculation on {len(dataset_paths)} datasets")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for dataset_path in tqdm(dataset_paths):
            flowrate_future = pool.submit(
                get_all_flowrates_from_experiment,
                top_level_dir=dataset_path.root_dir,
                verbose=False,
            )
            autofocus_future = pool.submit(
                autofocus_infer.predict,
                path_to_pth=args.path_to_autofocus_pth,
                path_to_zarr=dataset_path.zarr_path,
                device=next(devices),
            )
            yogo_future = pool.submit(
                yogo_infer.predict,
                path_to_pth=args.path_to_yogo_pth,
                path_to_zarr=dataset_path.zarr_path,
                device=next(devices),
            )

            dataset_path_dir = args.output_dir / dataset_path.root_dir.name
            dataset_path_dir.mkdir(exist_ok=True, parents=True)

            # while those are working, write the meta.yml file
            write_metadata_for_dataset_path(
                output_dir=dataset_path_dir,
                autofocus_path_to_pth=args.path_to_autofocus_pth,
                yogo_path_to_pth=args.path_to_yogo_pth,
            )

            wait(
                [flowrate_future, autofocus_future, yogo_future],
                return_when=ALL_COMPLETED,
            )

            # we could submit the writing job to the pool too!
            # but, it is actually pretty fast to write the results,
            # and this way we can check for errors immediately
            try:
                flowrate_results = flowrate_future.result()
                autofocus_results = autofocus_future.result()
                yogo_results = yogo_future.result()
            except Exception as e:
                print(f"error calculating results for {dataset_path_dir.name}: {e}")
                traceback.print_exc()
            else:
                write_results(
                    output_dir=dataset_path_dir,
                    flowrate_results=flowrate_results,
                    autofocus_results=autofocus_results,
                    yogo_results=yogo_results,
                    threshold_class_probabilities=False,
                )
