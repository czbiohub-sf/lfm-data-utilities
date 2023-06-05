#! /usr/bin/env python3

import warnings
import argparse

import torch

import matplotlib.pyplot as plt

from ruamel import yaml
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError

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
    tot_class_sum = torch.zeros(len(YOGO_CLASS_ORDERING), dtype=torch.long)
    for i, pred_slice in enumerate(predictions):
        pred = format_preds(pred_slice)
        if pred.numel() == 0:
            continue  # ignore no predictions
        classes = pred[:, 5:]
        class_predictions = classes.argmax(dim=1)
        tot_class_sum += torch.nn.functional.one_hot(
            class_predictions, num_classes=len(YOGO_CLASS_ORDERING)
        ).sum(dim=0)
    return tot_class_sum.squeeze()


def process_prediction(
    predictions: torch.Tensor,
    titration_point: str,
    result_dict: Dict[str, torch.Tensor],
) -> None:
    result_dict[titration_point] = get_prediction_class_counts(predictions)


def check_for_exceptions(futs: List[Future]):
    for fut in futs:
        try:
            maybe_exc = fut.exception(timeout=0.001)
        except TimeoutError:
            pass
        else:
            if maybe_exc is not None:
                raise maybe_exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("calcualte a titration curve")
    parser.add_argument("path_to_pth", type=Path, help="path to yogo pth file")
    parser.add_argument(
        "path_to_titration_yml",
        type=Path,
        help="path to a titration dataset description file",
    )
    parser.add_argument(
        "--plot-name",
        type=Path,
        help="name for the resulting plot",
        default=Path("./titration_plot.png"),
    )
    args = parser.parse_args()

    if torch.multiprocessing.cpu_count() < 32 or not torch.cuda.is_available():
        warnings.warn(
            "for best performance, we suggest running this script with 32 cpus "
            "and a gpu"
        )

    titration_points = load_titration_yml(args.path_to_titration_yml)
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
            vertical_crop_height_px=round(772 * 0.25),
        )
        # and process results asynchronously
        fut = tpe.submit(process_prediction, tn, titration_point, titration_results)
        futs.append(fut)

    with utils.timing_context_manager(
        "waiting for yogo prediction processing to complete"
    ):
        tpe.shutdown(wait=True)
        check_for_exceptions(futs)

    datapoints = sorted(titration_results.items())
    points, counts = zip(*datapoints)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(
        f"{args.path_to_pth.parent} titration on {args.path_to_titration_yml}",
        fontsize=16,
    )

    ax[0].set_title("Total number of cells per titration point")
    ax[0].set_xlabel("Titration point")
    ax[0].set_ylabel("Number of cells")
    ax[0].plot(points, [c.sum().item() for c in titration_results.values()])

    ax[1].set_title("Number of cells per class per titration point")
    ax[1].set_xlabel("Titration point")
    ax[1].set_ylabel("Number of cells")
    for i, class_name in enumerate(YOGO_CLASS_ORDERING):
        ax[1].plot(
            points, [c[i].item() for c in titration_results.values()], label=class_name
        )
    ax[1].legend()

    plt.savefig(f"{args.plot_name.with_suffix('.png')}")
