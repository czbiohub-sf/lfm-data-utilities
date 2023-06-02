#! /usr/bin/env python3

import argparse

import torch

import matplotlib.pyplot as plt

from ruamel import yaml
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, Future

from yogo.infer import predict
from yogo.utils import format_preds
from yogo.data.dataset import YOGO_CLASS_ORDERING

from lfm_data_utilities import utils


def load_titration_yml(path_to_titration_yml: Path) -> Dict[str, Path]:
    point_to_path: Dict[str, Path] = dict()
    with open(path_to_titration_yml, "r") as f:
        yaml_data = yaml.safe_load(f)
        points: Dict[str, str] = yaml_data["titration-points"]

        for titration_point, path in points.items():
            tpoint_path = Path(path)
            if not tpoint_path.exists():
                raise ValueError(
                    f"Path {path} for titration point {titration_point} does not exist"
                )
            point_to_path[titration_point] = tpoint_path

    return point_to_path


def get_prediction_class_counts(predictions: torch.Tensor) -> torch.Tensor:
    tot_class_sum = torch.zeros(1, len(YOGO_CLASS_ORDERING))
    for i, pred in enumerate(predictions):
        pred = format_preds(pred)
        classes = pred[:, 5:]
        tot_class_sum += classes.sum(dim=0)
    return tot_class_sum


def process_prediction(
    predictions: torch.Tensor,
    titration_point: str,
    result_dict: Dict[str, torch.Tensor],
) -> None:
    class_counts = get_prediction_class_counts(predictions)
    result_dict[titration_point] = class_counts


def check_for_exceptions(futs: List[Future]):
    for fut in futs:
        try:
            maybe_exc = fut.exception(timeout=0.001)
        except TimeoutError:
            pass
        if maybe_exc is not None:
            raise maybe_exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("calcualte a titration curve")
    parser.add_argument(
        "path_to_titration_yml", help="path to a titration dataset description file"
    )
    parser.add_argument("path_to_pth", help="path to yogo pth file")
    args = parser.parse_args()

    titration_points = load_titration_yml(Path(args.path_to_titration_yml))
    titration_results: Dict[str, torch.Tensor] = {}

    futs: List[Future] = []
    tpe = ThreadPoolExecutor(max_workers=4)
    for titration_point, path in titration_points.items():
        # let's error out asap
        check_for_exceptions(futs)
        # predict synchronously
        tn = predict(
            path_to_pth=args.path_to_pth,
            path_to_images=path,
            batch_size=64,
            use_tqdm=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # and process results asynchronously
        fut = tpe.submit(process_prediction, tn, titration_point, titration_results)
        futs.append(fut)

    with utils.timing_context_manager(
        "waiting for yogo prediction processing to complete"
    ):
        tpe.shutdown(wait=True)
        check_for_exceptions(futs)

    datapoints = sorted(titration_results.keys())
    points, counts = zip(*datapoints)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    ax[0].set_title("Total number of cells per titration point")
    ax[0].set_xlabel("Titration point")
    ax[0].set_ylabel("Number of cells")
    ax[0].scatter(points, [c.sum().item() for c in titration_results.values()])

    ax[1].set_title("Number of cells per class per titration point")
    ax[1].set_xlabel("Titration point")
    ax[1].set_ylabel("Number of cells")
    for i, class_name in enumerate(YOGO_CLASS_ORDERING):
        ax[1].scatter(
            points, [c[i].item() for c in titration_results.values()], label=class_name
        )
    ax[1].legend()

    plt.show()
