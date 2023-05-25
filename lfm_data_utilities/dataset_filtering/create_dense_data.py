#! /usr/bin/env python3

""" Jobs for this file:

1 find all run folders (by zarr file)
2 run SSAF, YOGO, and flowrate on all images
3 save metadata in a yml file
4 save results in a csv file
"""

import csv
import argparse
import warnings
import traceback

from pathlib import Path
from dataclasses import dataclass
from itertools import cycle
from typing import List, Tuple, Dict, Optional, cast
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import torch
import pandas as pd

# import these so we can get the package version identifiers (commit or __version__)
import yogo
import autofocus

from yogo.model import YOGO
from yogo.yogo_loss import YOGOLoss

from autofocus import infer as autofocus_infer
from yogo import infer as yogo_infer

from tqdm import tqdm
from ruamel.yaml import YAML

from filepath_dataset import get_dataloader

from lfm_data_utilities import utils
from lfm_data_utilities.malaria_labelling.labelling_constants import CLASSES
from lfm_data_utilities.image_processing.flowrate_utils import (
    get_all_flowrates_from_experiment,
)


def load_experiment_metadata(experiment_csv_path: Path) -> Dict[str, str]:
    with open(experiment_csv_path, "r") as ecp:
        return next(csv.DictReader(ecp), dict())


def write_metadata_for_dataset_path(
    output_dir: Path,
    autofocus_path_to_pth: Path,
    yogo_path_to_pth: Path,
    experiment_metadata_path: Optional[Path],
) -> None:
    autofocus_package_id = utils.try_get_package_version_identifier(autofocus)
    yogo_package_id = utils.try_get_package_version_identifier(yogo)

    # write all the above to meta.yml in output_dir
    yaml = YAML()
    meta = {
        "autofocus_package_id": autofocus_package_id,
        "yogo_package_id": yogo_package_id,
        "autofocus_path_to_pth": str(autofocus_path_to_pth.absolute()),
        "yogo_path_to_pth": str(yogo_path_to_pth.absolute()),
        **(
            load_experiment_metadata(experiment_metadata_path)
            if experiment_metadata_path is not None
            else dict()
        ),
    }
    with open(output_dir / "meta.yml", "w") as f:
        yaml.dump(meta, f)


@dataclass
class YOGOFolderResult:
    predictions: torch.Tensor
    losses: Optional[torch.Tensor] = None
    iou_loss: Optional[torch.Tensor] = None
    objectnes_loss_no_obj: Optional[torch.Tensor] = None
    objectnes_loss_obj: Optional[torch.Tensor] = None
    classification_loss: Optional[torch.Tensor] = None


def calculate_yogo_prediction_summary(
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

    return cast(List[float], result.tolist())


def yogo_analysis(
    calculate_yogo_loss, path_to_pth, path_to_zarr, device="cpu"
) -> YOGOFolderResult:
    if calculate_yogo_loss:
        mdl, cfg = YOGO.from_pth(path_to_pth)
        mdl.to(device)
        Sx, Sy = mdl.get_grid_size()

        image_path = path_to_zarr.parent / "images"
        label_path = path_to_zarr.parent / "labels"
        dataloader = get_dataloader(
            image_path,
            label_path,
            normalize_images=cfg["normalize_images"],
            batch_size=1,
            Sx=Sx,
            Sy=Sy,
        )
        y_loss = YOGOLoss()

        results = torch.zeros((len(dataloader), len(CLASSES) + 5, Sy, Sx))
        losses = torch.zeros(len(dataloader))
        iou_loss = torch.zeros(len(dataloader))
        objectnes_loss_no_obj = torch.zeros(len(dataloader))
        objectnes_loss_obj = torch.zeros(len(dataloader))
        classification_loss = torch.zeros(len(dataloader))
        for i, (img, label, path) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
                pred = mdl(img)
                loss, per_component_loss = y_loss(pred, label)

                results[i : i + pred.shape[0], ...] = pred.cpu()
                losses[i] = loss.cpu()
                iou_loss[i] = per_component_loss["iou_loss"]
                objectnes_loss_no_obj[i] = per_component_loss["objectnes_loss_no_obj"]
                objectnes_loss_obj[i] = per_component_loss["objectnes_loss_obj"]
                classification_loss[i] = per_component_loss["classification_loss"]

        return YOGOFolderResult(
            predictions=results,
            losses=losses,
            iou_loss=iou_loss,
            objectnes_loss_no_obj=objectnes_loss_no_obj,
            objectnes_loss_obj=objectnes_loss_obj,
            classification_loss=classification_loss,
        )
    else:
        return YOGOFolderResult(
            predictions=yogo_infer.predict(
                path_to_pth=path_to_pth, path_to_zarr=path_to_zarr, device=device
            )
        )


def write_results(
    output_dir: Path,
    flowrate_results: Tuple[List[float], List[float], List[float]],
    autofocus_results: torch.Tensor,
    yogo_results: YOGOFolderResult,
    threshold_class_probabilities: bool = True,
    objectness_threshold: float = 0.5,
) -> None:
    """Write results to csv file

    columns are:
        img_idx flowrate_dx flowrate_dy flowrate_confidence focus *calculated_yogo_summary
    """
    # flowrate of 1st frame can't be calculated, so set to 0
    # flowrate_iterable = chain(((0, 0, 0),), zip(*flowrate_results))
    flowrate_dx = [0] + flowrate_results[0]
    flowrate_dy = [0] + flowrate_results[1]
    flowrate_confidence = [0] + flowrate_results[2]

    yogo_class_predictions = [
        calculate_yogo_prediction_summary(
            yogo_res,
            threshold_class_probabilities=threshold_class_probabilities,
            objectness_threshold=0.5,
        )
        for yogo_res in yogo_results.predictions
    ]
    (
        num_healthy,
        num_ring,
        num_troph,
        num_schiz,
        num_gametocyte,
        num_wbc,
        num_misc,
    ) = zip(*yogo_class_predictions)
    assert (
        len(num_healthy)
        == len(num_ring)
        == len(num_troph)
        == len(num_schiz)
        == len(num_gametocyte)
        == len(num_wbc)
        == len(num_misc)
    )

    df = pd.DataFrame(
        {
            "flowrate_dx": flowrate_dx,
            "flowrate_dy": flowrate_dy,
            "flowrate_confidence": flowrate_confidence,
            "autofocus": autofocus_results.tolist(),
            "healthy": num_healthy,
            "ring": num_ring,
            "trophozoite": num_troph,
            "schizont": num_schiz,
            "gametocyte": num_gametocyte,
            "wbc": num_wbc,
            "misc": num_misc,
            "loss": yogo_results.losses,
            "iou_loss": yogo_results.iou_loss,
            "objectnes_loss_no_obj": yogo_results.objectnes_loss_no_obj,
            "objectnes_loss_obj": yogo_results.objectnes_loss_obj,
            "classification_loss": yogo_results.classification_loss,
        },
        columns=[
            "flowrate_dx",
            "flowrate_dy",
            "flowrate_confidence",
            "autofocus",
            *CLASSES,
            "loss",
            "iou_loss",
            "objectnes_loss_no_obj",
            "objectnes_loss_obj",
            "classification_loss",
        ],
    )
    df.to_csv(output_dir / "data.csv", index_label="img_idx")


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
    parser.add_argument(
        "--calculate-yogo-loss",
        action="store_true",
        default=True,
        help=("calculate loss across dataset"),
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
                yogo_analysis,
                args.calculate_yogo_loss,
                path_to_pth=args.path_to_yogo_pth,
                path_to_zarr=dataset_path.zarr_path,
                device=next(devices),
            )

            dataset_path_dir = args.output_dir / dataset_path.root_dir.name
            dataset_path_dir.mkdir(exist_ok=True, parents=True)

            # while those are working, write the meta.yml file
            try:
                write_metadata_for_dataset_path(
                    output_dir=dataset_path_dir,
                    experiment_metadata_path=dataset_path.experiment_csv_path,
                    autofocus_path_to_pth=args.path_to_autofocus_pth,
                    yogo_path_to_pth=args.path_to_yogo_pth,
                )
            except:
                print(f"error writing metadata for {dataset_path_dir.name}")
                traceback.print_exc()

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
                    threshold_class_probabilities=not args.expected_class_probabilities,
                    objectness_threshold=args.objectness_threshold,
                )
